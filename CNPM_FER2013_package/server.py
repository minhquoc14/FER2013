"""
server.py — FastAPI backend cho FER Web Demo (5 phương pháp)
Chạy: python -m uvicorn server:app --reload --port 8000
"""

import io
import time
import numpy as np
import cv2
from PIL import Image
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from skimage.feature import local_binary_pattern
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="FER Demo API — 5 Methods")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Nhãn cảm xúc ──────────────────────────────────────────────────────────────
EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
EMOTION_EMOJI  = ["😠", "🤢", "😨", "😊", "😐", "😢", "😲"]


# ══════════════════════════════════════════════════════════════════════════════
# Model definitions cho Method 4 và 5
# ══════════════════════════════════════════════════════════════════════════════

# ── CBAM cho EfficientNet+CBAM (dùng ca/sa) ───────────────────────────────────
class ChannelAttentionEff(nn.Module):
    def __init__(self, C, ratio=16):
        super().__init__()
        mid = max(C // ratio, 8)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(C, mid, bias=False), nn.ReLU(inplace=True),
            nn.Linear(mid, C, bias=False),
        )
        self.sig = nn.Sigmoid()

    def forward(self, x):
        w = self.sig(self.mlp(self.avg(x)) + self.mlp(self.max(x)))
        return x * w.unsqueeze(-1).unsqueeze(-1)


class SpatialAttentionEff(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sig  = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return x * self.sig(self.conv(torch.cat([avg, mx], dim=1)))


class CBAMEff(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.ca = ChannelAttentionEff(C)
        self.sa = SpatialAttentionEff()

    def forward(self, x):
        return self.sa(self.ca(x))


class EfficientNetB2_CBAM(nn.Module):
    def __init__(self, num_classes=7, dropout=0.4, freeze_until=3):
        super().__init__()
        base = models.efficientnet_b2(
            weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1
        )
        self.features = base.features
        for i, blk in enumerate(self.features):
            if i <= freeze_until:
                for p in blk.parameters():
                    p.requires_grad = False
        self.cbam4 = CBAMEff(88)
        self.cbam5 = CBAMEff(120)
        self.cbam6 = CBAMEff(208)
        self.fuse  = nn.Sequential(
            nn.Conv2d(88 + 120 + 208, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True),
        )
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout)
        self.fc      = nn.Linear(512, num_classes)

    def forward(self, x):
        for i, blk in enumerate(self.features):
            x = blk(x)
            if i == 4: f4 = x
            if i == 5: f5 = x
            if i == 6: f6 = x
        f4 = self.cbam4(f4)
        f5 = self.cbam5(f5)
        f6 = self.cbam6(f6)
        t  = (f6.shape[2], f6.shape[3])
        f4 = F.interpolate(f4, size=t, mode="bilinear", align_corners=False)
        f5 = F.interpolate(f5, size=t, mode="bilinear", align_corners=False)
        out = self.fuse(torch.cat([f4, f5, f6], dim=1))
        out = self.pool(out).flatten(1)
        return self.fc(self.dropout(out))


# ── CBAM cho ConvNeXt+CBAM (dùng channel/spatial) ────────────────────────────
class ChannelAttentionCnx(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg  = self.mlp(self.avg_pool(x))
        max_ = self.mlp(self.max_pool(x))
        return x * self.sigmoid(avg + max_).unsqueeze(-1).unsqueeze(-1)


class SpatialAttentionCnx(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size,
            padding=kernel_size // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        max_, _ = x.max(dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))


class CBAMCnx(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel = ChannelAttentionCnx(in_channels, reduction)
        self.spatial = SpatialAttentionCnx(kernel_size)

    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)
        return x


class ConvNeXtCBAM(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        convnext = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.DEFAULT
        )
        self.backbone = convnext.features
        self.cbam = CBAMCnx(in_channels=768, reduction=16, kernel_size=7)
        self.gap  = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(768),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.cbam(x)
        x = self.gap(x)
        return self.classifier(x)


# ══════════════════════════════════════════════════════════════════════════════
# Face Detector
# ══════════════════════════════════════════════════════════════════════════════
print("Loading face detector...")
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
print("  ✓ Face detector loaded")


# ══════════════════════════════════════════════════════════════════════════════
# Load Models
# ══════════════════════════════════════════════════════════════════════════════
print("Loading models...")

# Method 1: LBP + SVM
try:
    svm_model = joblib.load("saved_models/model_lbp_svm.pkl")
    print("  ✓ LBP + SVM loaded")
except Exception as e:
    svm_model = None
    print(f"  ✗ LBP + SVM: {e}")

# Method 2: CNN
try:
    cnn_model = torch.jit.load("saved_models/model_cnn_scripted.pt", map_location="cpu")
    cnn_model.eval()
    print("  ✓ CNN loaded")
except Exception as e:
    cnn_model = None
    print(f"  ✗ CNN: {e}")

# Method 3: EfficientNet-B2
try:
    pretrained_model = torch.jit.load("saved_models/model_pretrained_scripted.pt", map_location="cpu")
    pretrained_model.eval()
    print("  ✓ EfficientNet-B2 loaded")
except Exception as e:
    pretrained_model = None
    print(f"  ✗ EfficientNet-B2: {e}")

# Method 4: EfficientNet-B2 + CBAM
try:
    cbam_model = EfficientNetB2_CBAM(num_classes=7, dropout=0.4).to("cpu")
    cbam_model.load_state_dict(
        torch.load("saved_models/model_cbam.pth", map_location="cpu")
    )
    cbam_model.eval()
    print("  ✓ EfficientNet-B2 + CBAM loaded")
except Exception as e:
    cbam_model = None
    print(f"  ✗ EfficientNet-B2 + CBAM: {e}")

# Method 5: ConvNeXt + CBAM
try:
    convnext_model = ConvNeXtCBAM(num_classes=7).to("cpu")
    convnext_model.load_state_dict(
        torch.load("saved_models/model_convnext_cbam.pth", map_location="cpu")
    )
    convnext_model.eval()
    print("  ✓ ConvNeXt + CBAM loaded")
except Exception as e:
    convnext_model = None
    print(f"  ✗ ConvNeXt + CBAM: {e}")

print("Models loaded!\n")


# ══════════════════════════════════════════════════════════════════════════════
# Face Detection
# ══════════════════════════════════════════════════════════════════════════════
profile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)

def detect_and_crop_face(img: Image.Image):
    """Trả về (face_img, face_count)"""
    arr  = np.array(img.convert("RGB"))
    h_img, w_img = arr.shape[:2]

    # Ảnh nhỏ ≤ 100×100px → đã crop sẵn (FER2013, RAF-DB)
    if w_img <= 100 and h_img <= 100:
        return Image.fromarray(arr), 1

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # Bước 1: Frontal
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
    )
    # Bước 2: Profile nghiêng phải
    if len(faces) == 0:
        faces = profile_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
        )
    # Bước 3: Profile nghiêng trái
    if len(faces) == 0:
        gray_flip = cv2.flip(gray, 1)
        faces = profile_cascade.detectMultiScale(
            gray_flip, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
        )
        if len(faces) > 0:
            arr = cv2.flip(arr, 1)
    # Bước 4: scaleFactor nhỏ hơn
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20),
        )

    if len(faces) == 0:
        raise HTTPException(
            status_code=400,
            detail="Không phát hiện khuôn mặt. Vui lòng upload ảnh có khuôn mặt rõ ràng hơn."
        )

    face_count = len(faces)
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    pad_x = int(w * 0.2); pad_y = int(h * 0.2)
    x1 = max(0, x - pad_x); y1 = max(0, y - pad_y)
    x2 = min(arr.shape[1], x + w + pad_x)
    y2 = min(arr.shape[0], y + h + pad_y)
    return Image.fromarray(arr[y1:y2, x1:x2]), face_count


# ══════════════════════════════════════════════════════════════════════════════
# Preprocessing
# ══════════════════════════════════════════════════════════════════════════════
def preprocess_lbp(img):
    img = img.convert("L").resize((48, 48))
    arr = np.array(img, dtype=np.uint8)
    lbp = local_binary_pattern(arr, 16, 2, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=18, range=(0, 18), density=True)
    return hist.reshape(1, -1)

def preprocess_cnn(img):
    tf = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507], std=[0.255]),
    ])
    return tf(img).unsqueeze(0)

def preprocess_224(img):
    """Dùng chung cho Method 3, 4, 5 — đều 224×224 RGB"""
    tf = transforms.Compose([
        transforms.Grayscale(3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return tf(img).unsqueeze(0)


# ══════════════════════════════════════════════════════════════════════════════
# Predict helpers
# ══════════════════════════════════════════════════════════════════════════════
def predict_lbp(img):
    if svm_model is None:
        raise HTTPException(500, "LBP+SVM model chưa load")
    t0   = time.time()
    feat = preprocess_lbp(img)
    pred = svm_model.predict(feat)[0]
    try:
        scores = svm_model.decision_function(feat)[0]
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        scores = scores / scores.sum()
    except:
        scores = np.zeros(7); scores[pred] = 1.0
    return int(pred), scores.tolist(), round((time.time()-t0)*1000, 1)

def predict_cnn(img):
    if cnn_model is None:
        raise HTTPException(500, "CNN model chưa load")
    t0 = time.time()
    x  = preprocess_cnn(img)
    with torch.no_grad():
        probs = torch.softmax(cnn_model(x)[0], dim=0).numpy()
    return int(probs.argmax()), probs.tolist(), round((time.time()-t0)*1000, 1)

def predict_pretrained(img):
    if pretrained_model is None:
        raise HTTPException(500, "EfficientNet-B2 model chưa load")
    t0 = time.time()
    x  = preprocess_224(img)
    with torch.no_grad():
        probs = torch.softmax(pretrained_model(x)[0], dim=0).numpy()
    return int(probs.argmax()), probs.tolist(), round((time.time()-t0)*1000, 1)

def predict_cbam(img):
    if cbam_model is None:
        raise HTTPException(500, "EfficientNet-B2+CBAM model chưa load")
    t0 = time.time()
    x  = preprocess_224(img)
    with torch.no_grad():
        probs = torch.softmax(cbam_model(x)[0], dim=0).numpy()
    return int(probs.argmax()), probs.tolist(), round((time.time()-t0)*1000, 1)

def predict_convnext(img):
    if convnext_model is None:
        raise HTTPException(500, "ConvNeXt+CBAM model chưa load")
    t0 = time.time()
    x  = preprocess_224(img)
    with torch.no_grad():
        probs = torch.softmax(convnext_model(x)[0], dim=0).numpy()
    return int(probs.argmax()), probs.tolist(), round((time.time()-t0)*1000, 1)


# ── Helper tạo response ────────────────────────────────────────────────────────
def make_response(pred, probs, ms, method_name, accuracy, face_count=1):
    return {
        "method":    method_name,
        "emotion":   EMOTION_LABELS[pred],
        "emoji":     EMOTION_EMOJI[pred],
        "confidence": round(float(probs[pred]) * 100, 1),
        "probabilities": [
            {"label": EMOTION_LABELS[i], "emoji": EMOTION_EMOJI[i],
             "prob": round(float(p) * 100, 1)}
            for i, p in enumerate(probs)
        ],
        "time_ms":   ms,
        "model_acc": accuracy,
        "face_count": face_count,
    }


# ══════════════════════════════════════════════════════════════════════════════
# API Endpoints
# ══════════════════════════════════════════════════════════════════════════════
METHODS = {
    "lbp":        ("LBP + SVM",              28.00, predict_lbp),
    "cnn":        ("CNN",                    63.49, predict_cnn),
    "pretrained": ("EfficientNet-B2",        66.38, predict_pretrained),
    "cbam":       ("EfficientNet-B2 + CBAM", 67.05, predict_cbam),
    "convnext":   ("ConvNeXt + CBAM",        67.47, predict_convnext),
}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {
            "lbp_svm":        svm_model is not None,
            "cnn":            cnn_model is not None,
            "efficientnet":   pretrained_model is not None,
            "efficientnet_cbam": cbam_model is not None,
            "convnext_cbam":  convnext_model is not None,
        }
    }


@app.post("/predict/{method}")
async def predict(method: str, file: UploadFile = File(...)):
    if method not in METHODS:
        raise HTTPException(400, f"method phải là: {', '.join(METHODS.keys())}")
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except:
        raise HTTPException(400, "File không phải ảnh hợp lệ")
    face_img, face_count = detect_and_crop_face(img)
    name, acc, predict_fn = METHODS[method]
    pred, probs, ms = predict_fn(face_img)
    return make_response(pred, probs, ms, name, acc, face_count)


@app.post("/predict/all")
async def predict_all(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except:
        raise HTTPException(400, "File không phải ảnh hợp lệ")
    face_img, face_count = detect_and_crop_face(img)
    results  = []
    for method_key, (name, acc, predict_fn) in METHODS.items():
        try:
            pred, probs, ms = predict_fn(face_img)
            results.append({
                **make_response(pred, probs, ms, name, acc, face_count),
                "error": None,
            })
        except Exception as e:
            results.append({"method": name, "error": str(e)})
    return {"results": results}


# ── Serve frontend ─────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
def index():
    return FileResponse("frontend/index.html")