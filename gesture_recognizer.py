"""MediaPipe Gesture Recognizer 실시간 손 제스처 인식.

인식 가능한 제스처 (기본 모델 7종 + 없음):
    Closed_Fist ✊, Open_Palm ✋, Pointing_Up ☝️, Thumb_Down 👎,
    Thumb_Up 👍, Victory ✌️, ILoveYou 🤟, None(해당 없음)

실행:
    python gesture_recognizer.py        # 웹캠 0번
    python gesture_recognizer.py 1      # 웹캠 1번
    python gesture_recognizer.py video.mp4
    python gesture_recognizer.py "https://www.youtube.com/shorts/PWGWHPTTJ80"

종료: 창을 클릭한 뒤 q / ESC 키, 또는 창의 X 버튼
"""

import sys
import time

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

from hand_landmarker import download_youtube, draw_hands  # 영상 다운로드 / 뼈대 그리기 재사용

MODEL_PATH = "gesture_recognizer.task"  # 손 랜드마크 + 제스처 분류 모델 번들
DEFAULT_SOURCE = "0"
NUM_HANDS = 2
WINDOW_NAME = "MediaPipe Gesture Recognizer"


def should_quit():
    """q/Q/ESC 키를 누르거나 창의 X 버튼으로 창을 닫으면 True."""
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), ord("Q"), 27):
        return True
    return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1


def draw_gestures(frame, result):
    """손마다 인식된 제스처 이름과 점수를 손 위쪽에 표시한다."""
    h, w = frame.shape[:2]
    for landmarks, gestures in zip(result.hand_landmarks, result.gestures):
        top = gestures[0]  # 점수가 가장 높은 제스처
        x = int(min(lm.x for lm in landmarks) * w)
        y = int(min(lm.y for lm in landmarks) * h) - 15
        cv2.putText(frame, f"{top.category_name} {top.score:.2f}", (x, max(y, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SOURCE
    if source.isdigit():
        source = int(source)
    elif source.startswith("http"):
        source = download_youtube(source)

    options = vision.GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=NUM_HANDS,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"소스를 열 수 없습니다: {source}")
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    start = time.time()
    prev_time = start
    try:
        with vision.GestureRecognizer.create_from_options(options) as recognizer:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int((time.time() - start) * 1000)
                result = recognizer.recognize_for_video(mp_image, timestamp_ms)

                draw_hands(frame, result)
                draw_gestures(frame, result)

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
