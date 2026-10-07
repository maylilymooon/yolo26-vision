# YOLO26 · MediaPipe로 배우는 컴퓨터 비전 입문

> 강의 노트 · 2026-10-07
> 주제: Ultralytics YOLO26을 이용한 객체 탐지 · 세그멘테이션 · 객체 추적
> + MediaPipe를 이용한 손 랜드마크 · 제스처 인식 · 얼굴 AR 필터 · 나만의 제스처 학습

---

## 오늘의 학습 목표

이 강의를 마치면 다음을 할 수 있습니다.

1. Windows에서 Python이 "설치돼 있는데 실행이 안 되는" 문제를 진단하고 해결한다.
2. Ultralytics 라이브러리로 YOLO26 모델을 불러와 실행한다.
3. 같은 코드 구조로 **객체 탐지(Detection)**, **세그멘테이션(Segmentation)**, **객체 추적(Tracking)** 을 구현한다.
4. 웹캠, 동영상 파일, 유튜브 영상을 입력으로 바꿔 가며 테스트한다.
5. 완성한 코드를 GitHub에 올린다.
6. MediaPipe Tasks로 **손 관절**, **손 제스처**, **얼굴 랜드마크** 를 다루고, 얼굴에 AR 필터를 합성한다.
7. 원하는 제스처를 **직접 모으고 학습** 해서 인식한다.

---

## 목차

- [0교시. 개발 환경 준비](#0교시-개발-환경-준비)
- [1교시. YOLO26과 세 가지 비전 과제](#1교시-yolo26과-세-가지-비전-과제)
- [2교시. 객체 탐지 (Object Detection)](#2교시-객체-탐지-object-detection)
- [3교시. 입력 소스 바꾸기 (웹캠 없이 테스트하기)](#3교시-입력-소스-바꾸기-웹캠-없이-테스트하기)
- [4교시. 트러블슈팅: q를 눌러도 창이 안 꺼져요](#4교시-트러블슈팅-q를-눌러도-창이-안-꺼져요)
- [5교시. 세그멘테이션 (Segmentation)](#5교시-세그멘테이션-segmentation)
- [6교시. 객체 추적 (Object Tracking)](#6교시-객체-추적-object-tracking)
- [7교시. GitHub에 올리기](#7교시-github에-올리기)
- [8교시. MediaPipe 시작하기](#8교시-mediapipe-시작하기)
- [9교시. 손 랜드마크 (Hand Landmarker)](#9교시-손-랜드마크-hand-landmarker)
- [10교시. 제스처 인식 (Gesture Recognizer)](#10교시-제스처-인식-gesture-recognizer)
- [11교시. 얼굴 랜드마크로 AR 콧수염 필터 만들기](#11교시-얼굴-랜드마크로-ar-콧수염-필터-만들기)
- [12교시. 트러블슈팅: 유튜브 영상이 중간에 끊겨요](#12교시-트러블슈팅-유튜브-영상이-중간에-끊겨요)
- [13교시. 나만의 제스처 학습하기](#13교시-나만의-제스처-학습하기)
- [정리 및 복습 문제](#정리-및-복습-문제)

---

## 0교시. 개발 환경 준비

### 상황

`python`을 입력하면 "인식되지 않는 명령"이라는 오류가 납니다. 그런데 확인해 보니 **Python 3.13은 이미 설치돼 있었습니다.**

### 핵심 개념: PATH 환경 변수

> **PATH** 는 Windows가 명령어를 찾아볼 폴더들의 목록입니다.
> 프로그램이 컴퓨터에 있어도, 그 폴더가 PATH에 없으면 이름만으로는 실행할 수 없습니다.

### 진단 순서

```powershell
python --version          # 실패 → python 명령을 못 찾음
py --list                 # Python 런처로 설치된 버전 확인 → 3.13 있음!
```

Python은 `C:\Users\<사용자>\AppData\Local\Programs\Python\Python313\` 에 설치돼 있었습니다.

### 해결

사용자 PATH 맨 앞에 두 폴더를 추가했습니다.

| 폴더 | 역할 |
|---|---|
| `...\Python313\` | `python` 명령 |
| `...\Python313\Scripts\` | `pip` 등 Python 도구 명령 |

> ⚠️ **주의:** PATH를 바꾼 뒤에는 **터미널을 새로 열어야** 적용됩니다.

### 라이브러리 설치

```powershell
pip install -U ultralytics
```

PyTorch, OpenCV 등 필요한 패키지가 함께 설치됩니다.

---

## 1교시. YOLO26과 세 가지 비전 과제

**YOLO(You Only Look Once)** 는 이미지를 한 번만 보고 그 안의 객체를 빠르게 찾는 딥러닝 모델입니다. 오늘은 Ultralytics의 최신 버전인 **YOLO26** 을 사용합니다.

### 세 가지 과제 비교

| 과제 | 질문 | 출력 | 모델 파일 |
|---|---|---|---|
| **탐지** (Detection) | 무엇이 어디에 있나? | 사각형 상자 + 이름 + 신뢰도 | `yolo26n.pt` |
| **세그멘테이션** (Segmentation) | 정확히 어떤 픽셀이 그 객체인가? | 객체 모양 그대로의 마스크 | `yolo26n-seg.pt` |
| **추적** (Tracking) | 이 객체가 아까 그 객체인가? | 상자 + **고유 ID** | `yolo26n.pt` + 추적기 |

### 모델 크기

모델 이름의 `n` 자리를 바꾸면 크기가 달라집니다.

```
n (nano) → s (small) → m (medium) → l (large) → x (xlarge)
빠름 ←─────────────────────────────────────────→ 정확함
```

오늘은 일반 PC에서도 잘 돌아가는 `n` 모델을 씁니다.

> 💡 모델 파일(`.pt`)은 코드를 처음 실행할 때 **자동으로 다운로드** 됩니다.

---

## 2교시. 객체 탐지 (Object Detection)

📄 파일: [`webcam_detect.py`](webcam_detect.py)

### 전체 흐름

```
모델 불러오기 → 영상에서 프레임 하나 받기 → 탐지 → 결과 그리기 → 화면에 표시 → 반복
```

### 핵심 코드 해설

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")            # ① 모델 불러오기
```

```python
for result in model.predict(source, stream=True, conf=0.5, verbose=False):  # ②
    annotated = result.plot()         # ③ 결과 그림 만들기
    cv2.imshow(WINDOW_NAME, annotated)  # ④ 화면에 표시
```

| 번호 | 설명 |
|---|---|
| ① | 학습된 YOLO26 모델을 메모리에 올립니다. COCO 데이터셋의 **80가지 객체**(사람, 자동차, 개, 컵 등)를 알아봅니다. |
| ② | `predict()`가 영상의 프레임마다 탐지를 실행합니다. |
| ③ | `plot()`이 원본 프레임 위에 상자·이름·신뢰도를 그려 줍니다. |
| ④ | OpenCV로 결과 창에 띄웁니다. |

### 주요 옵션

| 옵션 | 의미 |
|---|---|
| `stream=True` | 결과를 한꺼번에 모으지 않고 **한 프레임씩** 돌려줍니다. 긴 영상에서도 메모리가 부족해지지 않습니다. |
| `conf=0.5` | 신뢰도가 50% 이상인 객체만 표시합니다. 낮추면 더 많이 찾지만 오탐도 늘어납니다. |
| `verbose=False` | 프레임마다 터미널에 로그를 찍지 않습니다. |

### FPS 표시

```python
fps = 1.0 / (지금 시각 - 이전 프레임 시각)
```

FPS(Frames Per Second)는 1초에 처리하는 프레임 수입니다. 높을수록 부드럽게 동작합니다.

---

## 3교시. 입력 소스 바꾸기 (웹캠 없이 테스트하기)

### 질문: 웹캠이 없으면 코드를 완성할 수 없나요?

**아니요.** 웹캠은 코드를 *작성* 할 때가 아니라 *실행해서 결과를 볼* 때만 필요합니다. 그리고 Ultralytics의 `source`에는 웹캠 말고도 여러 가지를 넣을 수 있습니다.

| `source` 값 | 입력 |
|---|---|
| `0`, `1`, ... | 웹캠 번호 |
| `"video.mp4"` | 동영상 파일 |
| `"photo.jpg"` | 사진 파일 |
| `"https://www.youtube.com/watch?v=..."` | 유튜브 영상 |
| `"http://192.168.0.5:8080/video"` | 휴대폰 IP 카메라 앱 |

### 명령줄 인자로 소스 받기

```python
source = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SOURCE
if source.isdigit():
    source = int(source)  # "0" 같은 숫자 문자열은 웹캠 번호로 변환
```

`sys.argv[1]`은 실행할 때 파일 이름 뒤에 적은 값입니다. 그래서 코드를 고치지 않고 명령만 바꿔서 입력을 바꿀 수 있습니다.

### 실습

```powershell
cd C:\Users\JBMOON\orca\projects\파이썬
python webcam_detect.py "https://www.youtube.com/watch?v=Ng5FYQUasgg"
```

> 💡 유튜브 주소는 `&` 같은 특수 문자가 들어갈 수 있으니 **따옴표로 감싸세요.**
> 💡 유튜브를 처음 입력하면 Ultralytics가 필요한 패키지를 자동으로 설치합니다.

---

## 4교시. 트러블슈팅: q를 눌러도 창이 안 꺼져요

실습 중 실제로 겪은 문제입니다.

### 원인 분석

`cv2.waitKey()`는 **결과 창에 들어온 키** 를 읽습니다. 그래서 다음 경우에는 `q`를 인식하지 못합니다.

| 원인 | 설명 |
|---|---|
| 한글 입력 상태 | `q` 자리를 누르면 `ㅂ`이 입력됩니다. |
| Caps Lock | 대문자 `Q`가 입력되는데, 처음 코드는 소문자 `q`만 확인했습니다. |
| 창 포커스 | 키 입력이 결과 창이 아니라 터미널로 갑니다. |

### 해결: 종료 조건을 함수로 분리

```python
def should_quit():
    """q/Q/ESC 키를 누르거나 창의 X 버튼으로 창을 닫으면 True."""
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), ord("Q"), 27):   # 27 = ESC
        return True
    return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
```

- **ESC(27)** 는 한/영 상태와 상관없이 같은 값이 들어옵니다.
- `getWindowProperty(..., WND_PROP_VISIBLE)`로 **X 버튼으로 창을 닫았는지** 도 확인합니다.

> 📌 **교훈:** 사용자 입력은 생각보다 다양합니다. 종료처럼 중요한 기능은 여러 방법을 열어 두세요.

---

## 5교시. 세그멘테이션 (Segmentation)

📄 파일: [`segment.py`](segment.py)

### 탐지와 무엇이 다른가?

탐지는 객체를 **사각형** 으로 감쌉니다. 세그멘테이션은 객체의 **실제 윤곽** 을 픽셀 단위로 칠합니다.

```
탐지:          세그멘테이션:
┌───────┐         ░░░
│  🧍   │        ░███░
│       │         ███
└───────┘         █ █
```

### 바뀐 코드는 딱 한 줄

```python
MODEL_PATH = "yolo26n-seg.pt"   # 이름 끝에 -seg
```

나머지 구조는 탐지 코드와 같습니다. `result.plot()`이 알아서 반투명 마스크까지 그려 줍니다.

### 추가한 기능: 객체 수 세기

```python
count = 0 if result.masks is None else len(result.masks)
```

`result.masks`에는 찾은 객체마다 하나씩 마스크가 들어 있습니다. 아무것도 못 찾으면 `None`이므로 먼저 확인합니다.

### 실습

```powershell
python segment.py "https://www.youtube.com/watch?v=Ng5FYQUasgg"
```

---

## 6교시. 객체 추적 (Object Tracking)

📄 파일: [`track.py`](track.py)

### 왜 추적이 필요한가?

탐지는 프레임마다 **처음부터 새로** 객체를 찾습니다. 그래서 "1초 전의 사람"과 "지금의 사람"이 같은 사람인지 모릅니다.
추적은 프레임 사이에서 객체를 **연결** 해서 같은 객체에 **같은 ID** 를 붙입니다.

```
프레임 1: person (id:1)   →   프레임 2: person (id:1)   →   프레임 3: person (id:1)
```

활용 예: 사람 수 세기, 이동 경로 분석, 차량 속도 측정

### 핵심 코드

```python
for result in model.track(source, stream=True, persist=True,
                          tracker="bytetrack.yaml", conf=0.5, verbose=False):
```

| 옵션 | 의미 |
|---|---|
| `model.track()` | `predict()` 대신 사용합니다. 탐지 + 추적을 함께 합니다. |
| `persist=True` | 프레임 사이에서 추적 정보를 **기억** 합니다. 이게 없으면 ID가 계속 새로 매겨집니다. |
| `tracker=` | 추적 알고리즘. `bytetrack.yaml`(빠름) 또는 `botsort.yaml`(ID 유지가 더 안정적인 편) |

### 이동 경로(Trail) 그리기

```python
trails = defaultdict(lambda: deque(maxlen=TRAIL_LENGTH))  # ID → 최근 중심점들

ids = result.boxes.id.int().tolist()           # 각 객체의 추적 ID
centers = result.boxes.xywh[:, :2].cpu().numpy()  # 각 상자의 중심 (x, y)
for track_id, (x, y) in zip(ids, centers):
    trails[track_id].append((int(x), int(y)))
    cv2.polylines(annotated, [points], isClosed=False, color=(0, 255, 255), thickness=3)
```

| 자료구조 | 쓰는 이유 |
|---|---|
| `defaultdict` | 처음 보는 ID가 나와도 자동으로 빈 기록을 만들어 줍니다. |
| `deque(maxlen=30)` | 30개가 넘으면 가장 오래된 점을 자동으로 버립니다. → 꼬리 길이가 일정하게 유지됩니다. |
| `set` (`seen_ids`) | 지금까지 등장한 서로 다른 ID의 개수를 셉니다. |

> ⚠️ 첫 프레임처럼 추적이 아직 시작되지 않으면 `result.boxes.id`가 `None`입니다. 그래서 `if result.boxes.id is not None:` 으로 먼저 확인합니다.

### 실습

```powershell
python track.py "https://www.youtube.com/watch?v=Ng5FYQUasgg"
```

---

## 7교시. GitHub에 올리기

### `.gitignore` — 올리지 않을 파일 정하기

```gitignore
__pycache__/
*.pt
*.task
videos/
gesture_data/
```

| 제외 대상 | 이유 |
|---|---|
| `__pycache__/` | Python이 자동으로 만드는 캐시 폴더 |
| `*.pt` | 모델 가중치 파일. 용량이 크고, 코드 실행 시 자동으로 다시 받을 수 있음 |
| `*.task` | MediaPipe 모델 번들. 공식 링크에서 다시 받을 수 있음 (8교시) |
| `videos/` | 테스트용으로 받은 유튜브 영상. 용량이 크고 저작권이 있음 |
| `gesture_data/` | 각자 모은 제스처 데이터와 학습한 모델 (13교시) |

> 📌 **원칙:** 저장소에는 **사람이 작성한 것** 만 올리고, 자동으로 생기거나 다시 받을 수 있는 파일은 올리지 않습니다.

### 업로드 명령

```powershell
git add .gitignore webcam_detect.py segment.py track.py
git commit -m "Add YOLO26 detection, segmentation and tracking scripts"
git branch -M main
gh repo create yolo26-vision --public --source . --remote origin --push
```

`gh repo create`는 GitHub 저장소 만들기, 원격 저장소 연결, 업로드를 한 번에 해 줍니다.

---

## 8교시. MediaPipe 시작하기

후반부는 Google의 **MediaPipe Tasks** 로 손과 얼굴을 다룹니다. YOLO가 "사람·자동차 같은 **객체**"를 찾는다면, MediaPipe는 손가락 마디·눈·입술 같은 **신체의 세부 지점(랜드마크)** 을 찾는 데 특화돼 있습니다.

### 라이브러리 설치

```powershell
pip install mediapipe             # MediaPipe Tasks (OpenCV contrib도 함께 설치됨)
pip install "yt-dlp[default]" deno imageio-ffmpeg   # 유튜브 영상을 받아 테스트할 때만 필요
```

| 패키지 | 역할 |
|---|---|
| `mediapipe` | 손·제스처·얼굴 인식 모델 실행 |
| `yt-dlp` | 유튜브 영상 다운로드 |
| `deno` | yt-dlp가 유튜브 페이지의 JavaScript를 풀 때 쓰는 런타임. 없으면 **HTTP 403** 으로 다운로드가 막힙니다. |
| `imageio-ffmpeg` | 영상 조각(HLS)을 하나의 mp4로 합칠 ffmpeg 실행 파일 |

### 모델 파일(`.task`) 받기

YOLO의 `.pt`와 달리 MediaPipe 모델은 **자동으로 받아지지 않습니다.** 공식 문서의 *Models* 표에서 `Latest` 링크를 눌러 받고, 코드와 같은 폴더에 둡니다. (이번 실습에서는 Claude in Chrome으로 받았습니다.)

| 과제 | 공식 문서 | 모델 파일 |
|---|---|---|
| 손 랜드마크 | [Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker) | [`hand_landmarker.task`](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task) (7.8MB) |
| 제스처 인식 | [Gesture Recognizer](https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer) | [`gesture_recognizer.task`](https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task) (8.4MB) |
| 얼굴 랜드마크 | [Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker) | [`face_landmarker.task`](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task) (3.8MB) |

```powershell
# 브라우저 대신 명령으로 받을 수도 있습니다
curl.exe -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

### 세 스크립트의 공통 구조

```python
options = vision.HandLandmarkerOptions(                       # ① 옵션 정하기
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
)
with vision.HandLandmarker.create_from_options(options) as landmarker:   # ② 모델 만들기
    while True:
        ok, frame = cap.read()                                # ③ OpenCV로 프레임 읽기
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)          # ④ BGR → RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)  # ⑤ 추론
        draw_hands(frame, result)                             # ⑥ 결과 직접 그리기
```

YOLO와 비교하면 다음이 다릅니다.

| | YOLO (Ultralytics) | MediaPipe Tasks |
|---|---|---|
| 영상 읽기 | `predict(source)`가 알아서 | **OpenCV `VideoCapture`로 직접** |
| 색 순서 | 신경 안 써도 됨 | OpenCV는 BGR, MediaPipe는 **RGB** → 변환 필수 |
| 결과 그리기 | `result.plot()` | **좌표를 받아 OpenCV로 직접** 그림 |
| 유튜브 입력 | 바로 가능 | 먼저 파일로 받아야 함 (`download_youtube()`) |

### `running_mode` 세 가지

| 모드 | 호출 함수 | 용도 |
|---|---|---|
| `IMAGE` | `detect(img)` | 사진 한 장씩 따로 |
| `VIDEO` | `detect_for_video(img, ms)` | 프레임을 순서대로. **이전 프레임의 위치로 추적** 해서 빠름 ← 오늘 사용 |
| `LIVE_STREAM` | `detect_async(img, ms)` + 콜백 | 결과를 비동기로 받음. 처리가 밀리면 프레임을 건너뜀 |

> ⚠️ `VIDEO` 모드의 타임스탬프(ms)는 **계속 커져야** 합니다. 같은 값이나 더 작은 값을 넣으면 오류가 납니다.

---

## 9교시. 손 랜드마크 (Hand Landmarker)

📄 파일: [`hand_landmarker.py`](hand_landmarker.py)

### 무엇을 알려주나?

손 하나마다 **21개의 관절점** 과 **왼손/오른손** 정보를 돌려줍니다.

| 번호 | 위치 |
|---|---|
| `0` | 손목 |
| `1`–`4` | 엄지 (4 = 끝) |
| `5`–`8` | 검지 (8 = 끝) |
| `9`–`12` | 중지 (12 = 끝) |
| `13`–`16` | 약지 (16 = 끝) |
| `17`–`20` | 새끼 (20 = 끝) |

손가락 끝(4, 8, 12, 16, 20)은 빨간 점, 나머지는 초록 점으로 그립니다.

### 정규화 좌표 → 픽셀 좌표

```python
points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
```

랜드마크의 `x`, `y`는 **0~1 사이 비율** 입니다. 화면 너비·높이를 곱해야 실제 픽셀 위치가 됩니다.

### 뼈대 그리기

```python
CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS   # (start, end) 쌍 목록
for conn in CONNECTIONS:
    cv2.line(frame, points[conn.start], points[conn.end], (255, 255, 255), 2)
```

> 💡 최신 MediaPipe(1.x)에는 예전의 `mp.solutions.drawing_utils`가 없습니다. 그래서 연결 정보만 받아 OpenCV로 직접 그립니다.

### 왼손/오른손이 반대로 나와요

MediaPipe는 **셀카처럼 좌우가 뒤집힌 영상** 을 기준으로 왼손/오른손을 판단합니다. 일반 영상(뒤집지 않은 영상)을 넣으면 `Left`/`Right`가 실제와 반대로 나옵니다. 웹캠에서는 `cv2.flip(frame, 1)`로 뒤집어 주면 맞게 나옵니다.

### 실습

```powershell
python hand_landmarker.py                                                 # 웹캠
python hand_landmarker.py "https://www.youtube.com/shorts/PWGWHPTTJ80"   # 손 쇼츠 영상
```

결과: 160프레임 전부에서 두 손 × 21개 관절이 검출됐습니다.

---

## 10교시. 제스처 인식 (Gesture Recognizer)

📄 파일: [`gesture_recognizer.py`](gesture_recognizer.py)

### 손 랜드마크 + 분류기

Gesture Recognizer는 9교시의 손 랜드마크 모델 **위에 분류 모델을 하나 더** 얹은 것입니다. 그래서 결과에 `hand_landmarks`, `handedness`가 그대로 있고, `gestures`가 추가됩니다.

| 제스처 | 이름 |
|---|---|
| ✊ | `Closed_Fist` |
| ✋ | `Open_Palm` |
| ☝️ | `Pointing_Up` |
| 👍 / 👎 | `Thumb_Up` / `Thumb_Down` |
| ✌️ | `Victory` |
| 🤟 | `ILoveYou` |
| (해당 없음) | `None` |

### 코드 재사용

```python
from hand_landmarker import download_youtube, draw_hands
```

결과의 구조가 같기 때문에 9교시의 `draw_hands()`를 그대로 가져다 씁니다. 새로 만든 것은 제스처 이름을 손 위에 쓰는 `draw_gestures()`뿐입니다.

```python
top = gestures[0]   # 점수가 가장 높은 제스처
cv2.putText(frame, f"{top.category_name} {top.score:.2f}", ...)
```

### 결과 해석하기

| 테스트 영상 | 결과 (손이 잡힐 때마다 1회) |
|---|---|
| [손 쇼츠](https://www.youtube.com/shorts/PWGWHPTTJ80) | `None` 257, `Open_Palm` 53 |
| [손 씻기 방송](https://www.youtube.com/watch?v=BMsHa1zIjK0) | `None` 4325, `Open_Palm` 717, `Thumb_Up` 108, `Closed_Fist` 9, `Thumb_Down` 2 |

손 씻기 영상에서는 물에 손을 대는 동작이 `Thumb_Up 0.51`로 잡히기도 했습니다.

> 📌 **교훈:** 모델은 **배운 7가지 중 하나로 억지로 분류** 하려고 합니다. 점수가 0.5 근처인 결과는 오인식일 가능성이 높으니, 실제 서비스에서는 점수 기준(예: 0.7 이상)을 두는 것이 좋습니다. 손 씻기 단계처럼 새로운 동작을 알아보게 하려면 **Model Maker로 직접 학습** 시켜야 합니다.

### 실습

```powershell
python gesture_recognizer.py          # 웹캠 앞에서 👍 ✌️ ✊ 해보기
python gesture_recognizer.py "https://www.youtube.com/watch?v=BMsHa1zIjK0"
```

---

## 11교시. 얼굴 랜드마크로 AR 콧수염 필터 만들기

📄 파일: [`face_mustache.py`](face_mustache.py)

### Face Landmarker가 주는 것

| 결과 | 내용 | 활용 예 |
|---|---|---|
| `face_landmarks` | 얼굴 **478개** 점 | AR 필터, 얼굴 정렬 |
| `face_blendshapes` | 표정 수치 **52개** (`eyeBlinkLeft`, `jawOpen`, `mouthSmileLeft` …) | 깜빡임·졸음 감지, 표정 인식 |
| `facial_transformation_matrixes` | 고개 회전·위치 | 고개 방향 추적 |

오늘은 랜드마크 478개 중 **4개만** 써서 콧수염을 붙입니다.

| 인덱스 | 위치 | 쓰임 |
|---|---|---|
| `2` | 코 밑 | 콧수염 **위치** |
| `0` | 윗입술 위쪽 | 콧수염 **위치** |
| `61`, `291` | 양쪽 입꼬리 | 콧수염 **크기** 와 **기울기** |

### ① 위치·크기·기울기 계산

```python
center = px(NOSE_BOTTOM) * 0.4 + px(UPPER_LIP_TOP) * 0.6   # 코 밑~윗입술 사이 (입술 쪽으로)
mouth_w = np.linalg.norm(right - left)                      # 입 너비
angle = math.degrees(math.atan2(right[1] - left[1], right[0] - left[0]))  # 입꼬리 선의 기울기
target_w = int(mouth_w * MUSTACHE_SCALE)                    # 콧수염 너비 = 입 너비 × 1.8
```

얼굴이 카메라에서 멀어지면 입 너비가 줄어 콧수염도 작아지고, 고개를 기울이면 입꼬리 선이 기울어 콧수염도 같이 기울어집니다.

### ② 콧수염 이미지를 코드로 그리기

이미지 파일을 따로 받지 않고 `make_mustache()`가 **투명 배경(BGRA)** 콧수염을 직접 그립니다.

```python
t = np.linspace(0, 1, 60)                                  # 0 = 가운데, 1 = 끝
mid = ... + np.sin(t * math.pi) * ... - t ** 4 * ...       # 중심선: 살짝 처졌다가 끝에서 위로
thick = height * 0.50 * (1 - t) ** 0.7 + ...               # 두께: 바깥으로 갈수록 얇게
```

오른쪽 절반의 윤곽을 계산한 뒤 **x를 좌우 대칭** 시켜 왼쪽을 만들고, 끝에 작은 원을 그려 말린 모양을 냅니다.

### ③ 투명 이미지를 회전시켜 합성하기 (알파 블렌딩)

```python
alpha = patch[:, :, 3:4] / 255.0                         # 0 = 투명, 1 = 불투명
roi[:] = patch[:, :, :3] * alpha + roi * (1 - alpha)     # 콧수염과 원본을 섞기
```

- 회전하면 모서리가 잘리므로, **대각선 길이의 정사각형 캔버스** 에 놓고 `cv2.warpAffine`으로 돌립니다.
- 얼굴이 화면 가장자리에 있으면 콧수염이 화면 밖으로 나가므로, **화면 안쪽 부분만 잘라서** 합칩니다.

### 실습

```powershell
python face_mustache.py                                                 # 웹캠
python face_mustache.py "https://www.youtube.com/watch?v=FPXPxtBOS7E"   # 인터뷰 영상
```

결과: 15분 영상 전체에서 고르게 뽑은 18장면 모두 얼굴이 잡혔고, 정면·옆모습 모두 인중에 콧수염이 붙었습니다.

### 연습 아이디어 (blendshape 활용)

| 난이도 | 예시 | 사용할 값 |
|---|---|---|
| ★ | 눈 깜빡임 카운터 / 졸음 경고 | `eyeBlinkLeft`, `eyeBlinkRight` |
| ★★ | 미소·입 벌림·놀람 표정 인식 | `mouthSmileLeft/Right`, `jawOpen`, `browInnerUp` |
| ★★ | 고개 방향(정면/좌/우) 표시 | `facial_transformation_matrixes` |

---

## 12교시. 트러블슈팅: 유튜브 영상이 중간에 끊겨요

MediaPipe 스크립트는 유튜브 주소를 받으면 `download_youtube()`로 `videos/` 폴더에 먼저 받아 둔 뒤 재생합니다. (한 번 받은 영상은 다시 받지 않습니다.) 이 과정에서 실제로 겪은 문제입니다.

| 증상 | 원인 | 해결 |
|---|---|---|
| `HTTP Error 403: Forbidden` | yt-dlp가 유튜브의 JavaScript를 풀지 못함 (`No supported JavaScript runtime` 경고) | `pip install "yt-dlp[default]" deno` |
| 15분 영상이 1분 6초에서 끝남 | 일반(DASH) 방식으로 받은 파일이 중간부터 깨짐 (`Invalid NAL unit size`) | **HLS(m3u8)** 방식을 우선해서 받기 |
| 720p를 골랐는데 1920×1080으로 읽힘 | 위와 같은 깨진 파일 | 위와 같음 |

```python
video = "bestvideo[ext=mp4][vcodec^=avc1][width<=1280][height<=1280]"
opts = {
    "format": f"{video}[protocol^=m3u8]/{video}/best[ext=mp4]/best",   # HLS 우선
    "ffmpeg_location": imageio_ffmpeg.get_ffmpeg_exe(),                # 조각 합치기용
}
```

| 조건 | 의미 |
|---|---|
| `ext=mp4`, `vcodec^=avc1` | OpenCV가 잘 읽는 **H.264** 영상 |
| `width<=1280`, `height<=1280` | 가로 영상은 720p, 세로 쇼츠는 720×1280까지. 용량과 처리 속도를 아낍니다. |
| `bestvideo` | 소리는 받지 않습니다. 검출에는 필요 없으니까요. |

> 📌 **교훈:** "실행은 되는데 결과가 이상하다"면 **입력 데이터부터 의심** 하세요. 이번에도 코드가 아니라 받은 영상 파일이 깨진 것이 원인이었습니다. `ffmpeg -i 파일 -c copy -f null -`로 프레임 수를 세어 보면 파일이 온전한지 확인할 수 있습니다.

---

## 13교시. 나만의 제스처 학습하기

📄 파일: [`gesture_collect.py`](gesture_collect.py) · [`gesture_train.py`](gesture_train.py) · [`gesture_custom.py`](gesture_custom.py) · [`custom_gesture_common.py`](custom_gesture_common.py)

10교시의 Gesture Recognizer는 7가지 제스처만 압니다. 이번에는 **내가 정한 제스처** 를 모으고, 학습하고, 인식까지 해 봅니다.

### 왜 Model Maker를 쓰지 않았나?

공식 방법인 **MediaPipe Model Maker** 는 TensorFlow 2.15가 필요한데, Python 3.13 + Windows에서는 설치되지 않습니다. 그래서 더 가벼운 방법을 씁니다.

```
손 영상 ─▶ Hand Landmarker(9교시) ─▶ 관절 21개 좌표 ─▶ 정규화(숫자 63개) ─▶ scikit-learn 분류기 ─▶ 제스처 이름
```

- **사진이 아니라 좌표만** 저장하므로 데이터가 작고, 학습이 몇 초면 끝납니다.
- 손을 찾는 어려운 일은 이미 학습된 MediaPipe가 하고, 우리는 **"이 좌표 모양이 어떤 제스처인가"** 만 학습합니다.

```powershell
pip install scikit-learn
```

### 핵심: 좌표 정규화

같은 제스처라도 손이 화면 어디에 있는지, 얼마나 큰지에 따라 좌표가 전부 달라집니다. 학습 전에 이 차이를 없앱니다.

```python
pts -= pts[0]                     # ① 손목(0번)을 원점으로 → 위치와 무관
pts /= max(손목에서 각 점까지 거리)   # ② 손 크기로 나누기 → 거리(크기)와 무관
if handedness == "Left":
    pts[:, 0] *= -1               # ③ 왼손은 좌우 반전 → 왼손·오른손을 같은 제스처로
```

> ⚠️ 이 함수(`landmarks_to_features`)는 **수집할 때와 추론할 때 똑같이** 써야 합니다. 그래서 `custom_gesture_common.py`에 한 번만 만들고 세 프로그램이 같이 씁니다.
> ⚠️ ③ 때문에 "왼쪽 가리키기"와 "오른쪽 가리키기"처럼 **방향만 다른 제스처는 구분되지 않습니다.**

### 1단계: 수집 (`python gesture_collect.py`)

Tkinter로 만든 창에서 영상을 보며 데이터를 모읍니다.

| 순서 | 할 일 |
|---|---|
| ① | **입력 소스** 에 `0`(웹캠) 또는 영상 파일·유튜브 주소를 넣고 **[열기]** |
| ② | **제스처 이름** 을 영문으로 입력하고 **[추가]** (예: `ok_sign`, `rock`, `call_me`, `none`) |
| ③ | 목록에서 제스처를 고르고 **● 녹화** 또는 **Space** |
| ④ | 손 모양을 유지하며 각도를 돌리고, 가까이·멀리 움직이기. 목표 개수(기본 200)가 차면 자동 정지 |
| ⑤ | 제스처마다 반복. 잘못 녹화했으면 **[마지막 녹화 되돌리기]** |

- 데이터는 `gesture_data/landmarks.csv`에 **라벨 + 숫자 63개** 한 줄씩 쌓입니다.
- 녹화 중에는 초당 10개만 저장합니다. 비슷한 프레임이 너무 많이 쌓이지 않게 하기 위해서입니다.
- 두 손이 같이 저장되면 라벨이 섞이므로, 수집할 때는 `num_hands=1`로 **한 손만** 봅니다.
- 이름은 영문·숫자·`_`만 됩니다. OpenCV 화면(`cv2.putText`)에 한글이 표시되지 않기 때문입니다.

> 💡 **`none` 제스처를 꼭 모으세요.** 아무 제스처도 아닌 평범한 손 모양을 모아 두면, 분류기가 아무 손에나 제스처 이름을 붙이는 일이 줄어듭니다.

### 2단계: 학습 (`python gesture_train.py`)

| 화면 | 내용 |
|---|---|
| 데이터 표 | 제스처별 샘플 수. 10개 미만이면 빨간색으로 표시되고 학습이 시작되지 않습니다. |
| 분류기 | **MLP 신경망(추천)** / 랜덤 포레스트 / KNN |
| 검증 비율 | 일부(기본 20%)를 떼어 두고 시험 문제로 씁니다. |
| 결과 | 검증 정확도, 제스처별 정밀도·재현율, **혼동 행렬** 그래프 |

```python
model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), ...))
```

- `StandardScaler`는 숫자 63개의 범위를 비슷하게 맞춰서 신경망이 잘 학습되게 합니다.
- 학습은 **별도 스레드** 에서 돌립니다. 그렇지 않으면 학습하는 동안 창이 "응답 없음"이 됩니다.
- 검증이 끝나면 **전체 데이터로 한 번 더 학습** 해서 `gesture_data/custom_gesture_model.joblib`로 저장합니다.

> 📌 **혼동 행렬 읽는 법:** 대각선(정답 = 예측)이 진할수록 좋습니다. 대각선 밖에 숫자가 있으면 그 두 제스처를 헷갈린 것이니, 그 둘의 데이터를 더 모으세요.

### 3단계: 인식 (`python gesture_custom.py`)

```powershell
python gesture_custom.py          # 웹캠 (거울 모드)
python gesture_custom.py "영상 또는 유튜브 주소"
```

```python
probs = model.predict_proba([feats])[0]
name = model.classes_[probs.argmax()] if probs.max() >= 0.6 else "Unknown"
```

확률이 0.6보다 낮으면 회색 `Unknown`으로 표시해서 애매한 손 모양에 엉뚱한 이름이 붙지 않게 합니다.

### 트러블슈팅: 유튜브 주소를 열었더니 너무 느려요

| 원인 | 해결 |
|---|---|
| 10분 영상을 720p(165MB)로 받는데 진행 표시가 없어서 멈춘 것처럼 보임 | 수집기는 **480p 이하** 로 받고(이 영상은 360p 21MB, 약 15초), 진행률 % 표시 |
| 기다리다 [열기]를 또 누르면 같은 파일을 동시에 받다가 `WinError 32`로 실패 | 받는 동안 **[열기] 버튼 잠금** |
| 이미 받은 영상도 열 때마다 유튜브에 정보를 물어봐서 몇 초 대기 | 주소에서 **영상 ID를 바로 뽑아** 받아 둔 파일이 있으면 즉시 열기 |
| 처리가 빨라서 영상 파일이 2배속으로 지나감 | 다음 프레임 시각을 누적 계산해서 **원래 속도(30fps)** 로 재생 |

> 💡 해상도를 낮추면 손이 덜 잡힙니다. 이 영상에서는 720p 64% → 360p 56%였습니다. 녹화 시간이 조금 늘어날 뿐 데이터 품질에는 문제가 없습니다.

---

## 정리 및 복습 문제

### 오늘 만든 파일

| 파일 | 과제 | 모델 | 핵심 함수 |
|---|---|---|---|
| `webcam_detect.py` | 객체 탐지 | `yolo26n.pt` | `model.predict()` |
| `segment.py` | 세그멘테이션 | `yolo26n-seg.pt` | `model.predict()` |
| `track.py` | 객체 추적 | `yolo26n.pt` | `model.track(persist=True)` |
| `hand_landmarker.py` | 손 랜드마크 (21점) | `hand_landmarker.task` | `detect_for_video()` |
| `gesture_recognizer.py` | 손 제스처 (7종) | `gesture_recognizer.task` | `recognize_for_video()` |
| `face_mustache.py` | 얼굴 AR 필터 | `face_landmarker.task` | `detect_for_video()` + 알파 블렌딩 |
| `gesture_collect.py` | 제스처 데이터 수집 (UI) | `hand_landmarker.task` | `landmarks_to_features()` |
| `gesture_train.py` | 제스처 분류기 학습 (UI) | → `custom_gesture_model.joblib` | `MLPClassifier.fit()` |
| `gesture_custom.py` | 나만의 제스처 인식 | `hand_landmarker.task` + `.joblib` | `predict_proba()` |

### 실행 방법 (공통)

```powershell
python <파일이름>.py                 # 웹캠 0번
python <파일이름>.py video.mp4       # 동영상 파일
python <파일이름>.py "유튜브 주소"     # 유튜브
```

종료: 결과 창 클릭 후 `q` / `ESC`, 또는 창의 X 버튼

### 핵심 요약

1. **PATH** 에 없으면 설치된 프로그램도 이름만으로 실행할 수 없다.
2. Ultralytics에서는 **모델 파일만 바꾸면** 탐지 ↔ 세그멘테이션을 전환할 수 있다.
3. **`predict()` → `track(persist=True)`** 로 바꾸면 추적이 된다.
4. `source`에는 웹캠, 파일, 유튜브 주소를 모두 넣을 수 있다.
5. 키 입력은 한글 상태, Caps Lock, 창 포커스의 영향을 받는다.
6. MediaPipe는 **RGB 입력**, **0~1 정규화 좌표** 를 쓰고, 결과는 직접 그려야 한다.
7. `VIDEO` 모드는 이전 프레임으로 추적해 빠르지만, 타임스탬프가 계속 커져야 한다.
8. 랜드마크 몇 개의 **위치·거리·각도** 만으로 AR 필터의 위치·크기·회전을 정할 수 있다.
9. 랜드마크 좌표를 **정규화** 하면 작은 분류기로도 나만의 제스처를 학습할 수 있다.

### 복습 문제

1. `python` 명령이 안 될 때, Python이 설치돼 있는지 확인하는 명령은 무엇인가요?
2. 세그멘테이션을 하려면 탐지 코드에서 무엇을 바꿔야 하나요?
3. `model.track()`에서 `persist=True`를 빼면 어떤 일이 생길까요?
4. `conf=0.5`를 `0.2`로 바꾸면 결과가 어떻게 달라질까요?
5. `deque(maxlen=30)` 대신 일반 리스트를 쓰면 어떤 문제가 생길까요?
6. `.gitignore`에 `*.pt`를 넣은 이유는 무엇인가요?
7. MediaPipe에 OpenCV 프레임을 넣기 전에 꼭 해야 하는 변환은 무엇인가요?
8. 일반 영상에서 오른손이 `Left`로 나오는 이유는 무엇인가요?
9. 콧수염의 크기와 기울기를 정할 때 입꼬리 두 점(61, 291)을 쓰는 이유는 무엇인가요?
10. 제스처 좌표를 학습하기 전에 손목을 원점으로 옮기고 손 크기로 나누는 이유는 무엇인가요?
11. 나만의 제스처를 모을 때 `none` 제스처를 함께 모으면 좋은 이유는 무엇인가요?

<details>
<summary>정답 보기</summary>

1. `py --list`
2. 모델 파일을 `yolo26n.pt` → `yolo26n-seg.pt` 로 바꾼다.
3. 프레임마다 추적 정보가 초기화되어, 같은 객체여도 ID가 유지되지 않는다.
4. 더 많은 객체를 찾지만, 잘못 찾은 객체(오탐)도 늘어난다.
5. 점이 끝없이 쌓여 메모리를 계속 쓰고, 꼬리가 화면 전체에 길게 남는다.
6. 모델 파일은 용량이 크고, 코드를 실행하면 자동으로 다시 받을 수 있기 때문이다.
7. `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)` — OpenCV는 BGR, MediaPipe는 RGB를 쓴다.
8. MediaPipe는 좌우가 뒤집힌(셀카) 영상을 기준으로 왼손/오른손을 판단하기 때문이다.
9. 두 점 사이 거리는 얼굴 크기에 비례하고, 두 점을 잇는 선의 각도는 고개 기울기를 따라가기 때문이다.
10. 손이 화면 어디에 있든, 카메라에서 얼마나 멀든 같은 제스처가 같은 숫자로 표현되게 하기 위해서다.
11. 분류기는 항상 아는 제스처 중 하나를 고르기 때문에, '아무것도 아님'을 따로 배워야 오인식이 줄어든다.

</details>

### 도전 과제

- [ ] `MODEL_PATH`를 `yolo26s.pt`로 바꿔서 FPS와 정확도 차이를 비교해 보기
- [ ] `TRACKER`를 `botsort.yaml`로 바꿔서 ID가 얼마나 잘 유지되는지 비교해 보기
- [ ] `track.py`에서 `person`만 추적하도록 바꿔 보기 (힌트: `predict`/`track`의 `classes=[0]` 옵션)
- [ ] 화면에 가로선을 긋고, 선을 넘어간 사람 수를 세어 보기
- [ ] `gesture_recognizer.py`에서 점수 0.7 미만과 `None`은 표시하지 않도록 바꿔 보기
- [ ] `hand_landmarker.py`에 검지 끝(8번)으로 화면에 그림 그리는 기능 넣어 보기
- [ ] Face Landmarker의 `eyeBlinkLeft/Right`로 눈 깜빡임 횟수 세기 (옵션: `output_face_blendshapes=True`)
- [ ] 콧수염 대신 선글라스 붙여 보기 (힌트: 눈 바깥쪽 33, 263번)
- [ ] 나만의 제스처 3개 + `none`을 모아 학습하고, 분류기 3종의 정확도 비교해 보기
- [ ] `gesture_custom.py`에서 특정 제스처가 나오면 스크린샷을 저장하도록 바꿔 보기
