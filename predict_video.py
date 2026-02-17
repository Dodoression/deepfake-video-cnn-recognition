import torch
import numpy as np
import cv2
from deepfake_baseline import (
    SmallCNN,
    sample_frames,
    crop_largest_face_bgr,
    build_haar_face_detector,
    IMG_SIZE,
    FPS_SAMPLE,
    USE_FACE_CROP,
    DEVICE
)

import torchvision.transforms as transforms

MODEL_PATH = "deepfake_smallcnn.pt"

tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
])

def predict_video(video_path):
    model = SmallCNN(num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    face_detector = build_haar_face_detector() if USE_FACE_CROP else None

    frames = sample_frames(video_path, FPS_SAMPLE)

    probs = []

    softmax = torch.nn.Softmax(dim=1)

    for frame in frames:
        if USE_FACE_CROP:
            cropped = crop_largest_face_bgr(frame, face_detector)
            if cropped is not None:
                frame = cropped

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        x = tf(frame).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(x)
            p = softmax(logits)[0].cpu().numpy()

        probs.append(p)

    mean_prob = np.mean(np.stack(probs, axis=0), axis=0)

    label = "FAKE" if np.argmax(mean_prob) == 1 else "REAL"

    return label, mean_prob


if __name__ == "__main__":
    video = "mytest.mp4"
    label, prob = predict_video(video)

    print("Prediction:", label)
    print("Probabilities [REAL, FAKE]:", prob)
