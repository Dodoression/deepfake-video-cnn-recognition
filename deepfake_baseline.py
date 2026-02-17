import os
import cv2
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# -----------------------------
# Config
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "data")          # data/real, data/fake
FPS_SAMPLE = 4              # frames per second to sample
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3
SEED = 42
USE_FACE_CROP = True        # easy baseline (Haar)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------------
# Utilities: frame sampling + (optional) face crop
# -----------------------------
def build_haar_face_detector():
    # Comes with OpenCV
    haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(haar_path):
        raise RuntimeError("Could not find haarcascade_frontalface_default.xml")
    return cv2.CascadeClassifier(haar_path)


def crop_largest_face_bgr(img_bgr, face_detector):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None

    # pick largest
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # expand a bit for context
    pad = int(0.2 * max(w, h))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(img_bgr.shape[1], x + w + pad)
    y1 = min(img_bgr.shape[0], y + h + pad)
    return img_bgr[y0:y1, x0:x1]


def sample_frames(video_path, fps_sample=2):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 25.0
    frame_interval = max(int(round(fps / fps_sample)), 1)

    frames = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % frame_interval == 0:
            frames.append(frame)
        idx += 1

    cap.release()
    return frames


# -----------------------------
# Dataset: returns frames with label
# -----------------------------
class VideoFramesDataset(Dataset):
    """
    Converts a list of videos into frame-level samples.
    Each sample: (tensor_image, label, video_id)
    """
    def __init__(self, video_paths, labels, img_size=224, fps_sample=2, use_face_crop=True):
        self.video_paths = video_paths
        self.labels = labels
        self.img_size = img_size
        self.fps_sample = fps_sample
        self.use_face_crop = use_face_crop

        self.face_detector = build_haar_face_detector() if use_face_crop else None

        self.samples = []  # list of (video_path, frame_index, label)
        self._index_videos()

        self.tf = transforms.Compose([
            transforms.ToTensor(),  # HWC [0..255] -> CHW [0..1]
            transforms.Resize((img_size, img_size)),
        ])

    def _index_videos(self):
        for vp, y in tqdm(list(zip(self.video_paths, self.labels)), desc="Indexing videos"):
            frames = sample_frames(vp, self.fps_sample)
            MAX_FRAMES_PER_VIDEO = 30
            frames = frames[:MAX_FRAMES_PER_VIDEO]
            for i in range(len(frames)):
                self.samples.append((vp, i, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        vp, frame_i, y = self.samples[idx]
        frames = sample_frames(vp, self.fps_sample)
        if frame_i >= len(frames):
            frame = frames[-1]
        else:
            frame = frames[frame_i]

        if self.use_face_crop:
            cropped = crop_largest_face_bgr(frame, self.face_detector)
            if cropped is not None:
                frame = cropped

        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # convert to tensor
        img = self.tf(frame)
        return img, torch.tensor(y, dtype=torch.long), os.path.basename(vp)


# -----------------------------
# Simple CNN (small, fast baseline)
# -----------------------------
class SmallCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.net(x)
        x = x.flatten(1)
        return self.fc(x)


# -----------------------------
# Train/Eval
# -----------------------------
def split_train_val(video_paths, labels, val_ratio=0.2):
    idxs = list(range(len(video_paths)))
    random.shuffle(idxs)
    n_val = int(len(idxs) * val_ratio)

    val_idxs = idxs[:n_val]
    train_idxs = idxs[n_val:]

    train_v = [video_paths[i] for i in train_idxs]
    train_y = [labels[i] for i in train_idxs]
    val_v = [video_paths[i] for i in val_idxs]
    val_y = [labels[i] for i in val_idxs]

    return train_v, train_y, val_v, val_y


@torch.no_grad()
def eval_frame_level(model, loader):
    model.eval()
    y_true, y_pred = [], []
    for x, y, _ in loader:
        x = x.to(DEVICE)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).cpu().numpy()
        y_true.extend(y.numpy())
        y_pred.extend(pred)
    return accuracy_score(y_true, y_pred), confusion_matrix(y_true, y_pred), classification_report(y_true, y_pred)


@torch.no_grad()
def eval_video_level(model, loader):
    """
    Aggregates frame predictions per video via mean probability.
    """
    model.eval()
    probs_by_video = {}
    y_by_video = {}

    softmax = nn.Softmax(dim=1)

    for x, y, vid in loader:
        x = x.to(DEVICE)
        logits = model(x)
        probs = softmax(logits).cpu().numpy()  # [B,2]
        y_np = y.numpy()
        vid = list(vid)

        for i in range(len(vid)):
            v = vid[i]
            if v not in probs_by_video:
                probs_by_video[v] = []
                y_by_video[v] = y_np[i]
            probs_by_video[v].append(probs[i])

    y_true, y_pred = [], []
    for v, plist in probs_by_video.items():
        mean_prob = np.mean(np.stack(plist, axis=0), axis=0)
        pred = int(np.argmax(mean_prob))
        y_true.append(int(y_by_video[v]))
        y_pred.append(pred)

    return accuracy_score(y_true, y_pred), confusion_matrix(y_true, y_pred), classification_report(y_true, y_pred)


def main():
    set_seed(SEED)

    real_dir = os.path.join(DATA_ROOT, "real")
    fake_dir = os.path.join(DATA_ROOT, "fake")

    real_videos = [os.path.join(real_dir, f) for f in os.listdir(real_dir) if f.lower().endswith(".mp4")]
    fake_videos = [os.path.join(fake_dir, f) for f in os.listdir(fake_dir) if f.lower().endswith(".mp4")]

    video_paths = real_videos + fake_videos
    labels = [0] * len(real_videos) + [1] * len(fake_videos)  # 0=REAL, 1=FAKE

    if len(video_paths) < 4:
        raise RuntimeError("Need at least a few videos in data/real and data/fake")

    train_v, train_y, val_v, val_y = split_train_val(video_paths, labels, val_ratio=0.2)

    train_ds = VideoFramesDataset(train_v, train_y, IMG_SIZE, FPS_SAMPLE, USE_FACE_CROP)
    val_ds = VideoFramesDataset(val_v, val_y, IMG_SIZE, FPS_SAMPLE, USE_FACE_CROP)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = SmallCNN(num_classes=2).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for x, y, _ in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(len(train_loader), 1)
        acc_f, cm_f, rep_f = eval_frame_level(model, val_loader)
        acc_v, cm_v, rep_v = eval_video_level(model, val_loader)

        print(f"\nEpoch {epoch} | loss={avg_loss:.4f}")
        print(f"[FRAME] acc={acc_f:.4f}\n{cm_f}\n{rep_f}")
        print(f"[VIDEO] acc={acc_v:.4f}\n{cm_v}\n{rep_v}")

    torch.save(model.state_dict(), "deepfake_smallcnn.pt")
    print("Saved model: deepfake_smallcnn.pt")


if __name__ == "__main__":
    main()
