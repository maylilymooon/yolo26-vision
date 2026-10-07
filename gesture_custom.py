"""직접 학습한 제스처 분류기로 실시간 추론.

gesture_collect.py → gesture_train.py 로 만든 gesture_data/custom_gesture_model.joblib 를 쓴다.

실행:
    python gesture_custom.py        # 웹캠 0번 (거울 모드)
    python gesture_custom.py 1
    python gesture_custom.py video.mp4
    python gesture_custom.py "https://www.youtube.com/watch?v=..."

종료: 창을 클릭한 뒤 q / ESC 키, 또는 창의 X 버튼
"""

import sys
import time

import cv2
import joblib
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from custom_gesture_common import HAND_MODEL_PATH, MODEL_OUT_PATH, landmarks_to_features
from hand_landmarker import download_youtube, draw_hands

DEFAULT_SOURCE = "0"
NUM_HANDS = 2
CONF_THRESHOLD = 0.6      # 이 확률보다 낮으면 "Unknown"으로 표시 (오인식 줄이기)
WINDOW_NAME = "Custom Gesture"


def should_quit():
    """q/Q/ESC 키를 누르거나 창의 X 버튼으로 창을 닫으면 True."""
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), ord("Q"), 27):
        return True
    return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SOURCE
    mirror = source.isdigit()  # 웹캠은 거울처럼 좌우 반전해서 보여주기
    if source.isdigit():
        source = int(source)
    elif source.startswith("http"):
        source = download_youtube(source)

    try:
        bundle = joblib.load(MODEL_OUT_PATH)
    except FileNotFoundError:
        sys.exit(f"모델이 없습니다: {MODEL_OUT_PATH}\n먼저 gesture_collect.py → gesture_train.py 를 실행하세요.")
    model = bundle["model"]
    print(f"모델: {bundle['classifier']} · 제스처 {bundle['labels']} · "
          f"검증 정확도 {bundle['val_accuracy']:.1%} ({bundle['trained_at']})")

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=NUM_HANDS,
    )

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"소스를 열 수 없습니다: {source}")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    start = time.time()
    prev_time = start
    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if mirror:
                    frame = cv2.flip(frame, 1)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                timestamp_ms = int((time.time() - start) * 1000)
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms)
                draw_hands(frame, result)

                h, w = frame.shape[:2]
                for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                    feats = landmarks_to_features(landmarks, handedness[0].category_name)
                    probs = model.predict_proba([feats])[0]
                    best = probs.argmax()
                    name = model.classes_[best] if probs[best] >= CONF_THRESHOLD else "Unknown"
                    x = int(min(lm.x for lm in landmarks) * w)
                    y = int(min(lm.y for lm in landmarks) * h) - 15
                    color = (0, 255, 255) if name != "Unknown" else (160, 160, 160)
                    cv2.putText(frame, f"{name} {probs[best]:.2f}", (x, max(y, 30)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

                now = time.time()
                fps = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow(WINDOW_NAME, frame)
                if should_quit():
                    break
            else:
                while not should_quit():
                    pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
