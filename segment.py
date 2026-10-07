"""Ultralytics YOLO26 실시간 인스턴스 세그멘테이션.

실행:
    python segment.py                     # 웹캠 0번
    python segment.py video.mp4           # 동영상/사진 파일
    python segment.py "https://www.youtube.com/watch?v=..."  # 유튜브
종료: 창을 클릭한 뒤 q / ESC 키, 또는 창의 X 버튼
"""

import sys
import time

import cv2
from ultralytics import YOLO

MODEL_PATH = "yolo26n-seg.pt"  # n(가장 빠름) / s / m / l / x(가장 정확), 이름 끝에 -seg
DEFAULT_SOURCE = "0"           # 인자가 없을 때 쓸 소스 (기본 웹캠)
CONF_THRESHOLD = 0.5           # 이 신뢰도 이상인 객체만 표시
WINDOW_NAME = "YOLO26 Segmentation"


def should_quit():
    """q/Q/ESC 키를 누르거나 창의 X 버튼으로 창을 닫으면 True."""
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), ord("Q"), 27):
        return True
    return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SOURCE
    if source.isdigit():
        source = int(source)  # 숫자는 웹캠 번호

    model = YOLO(MODEL_PATH)  # 처음 실행 시 가중치 자동 다운로드
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    prev_time = time.time()
    try:
        for result in model.predict(source, stream=True, conf=CONF_THRESHOLD, verbose=False):
            # 객체마다 반투명 색 마스크 + 상자 + 이름/신뢰도를 그림
            annotated = result.plot()

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now
            count = 0 if result.masks is None else len(result.masks)
            cv2.putText(annotated, f"FPS: {fps:.1f}  Objects: {count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow(WINDOW_NAME, annotated)
            if should_quit():
                break
        else:
            # 영상이 끝나거나 사진 한 장이면 종료 키를 누를 때까지 창 유지
            while not should_quit():
                pass
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
