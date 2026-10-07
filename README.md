# YOLO26으로 배우는 컴퓨터 비전 입문

> 강의 노트 · 2026-10-07
> 주제: Ultralytics YOLO26을 이용한 객체 탐지 · 세그멘테이션 · 객체 추적

---

## 오늘의 학습 목표

이 강의를 마치면 다음을 할 수 있습니다.

1. Windows에서 Python이 "설치돼 있는데 실행이 안 되는" 문제를 진단하고 해결한다.
2. Ultralytics 라이브러리로 YOLO26 모델을 불러와 실행한다.
3. 같은 코드 구조로 **객체 탐지(Detection)**, **세그멘테이션(Segmentation)**, **객체 추적(Tracking)** 을 구현한다.
4. 웹캠, 동영상 파일, 유튜브 영상을 입력으로 바꿔 가며 테스트한다.
5. 완성한 코드를 GitHub에 올린다.

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
```

| 제외 대상 | 이유 |
|---|---|
| `__pycache__/` | Python이 자동으로 만드는 캐시 폴더 |
| `*.pt` | 모델 가중치 파일. 용량이 크고, 코드 실행 시 자동으로 다시 받을 수 있음 |

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

## 정리 및 복습 문제

### 오늘 만든 파일

| 파일 | 과제 | 모델 | 핵심 함수 |
|---|---|---|---|
| `webcam_detect.py` | 객체 탐지 | `yolo26n.pt` | `model.predict()` |
| `segment.py` | 세그멘테이션 | `yolo26n-seg.pt` | `model.predict()` |
| `track.py` | 객체 추적 | `yolo26n.pt` | `model.track(persist=True)` |

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

### 복습 문제

1. `python` 명령이 안 될 때, Python이 설치돼 있는지 확인하는 명령은 무엇인가요?
2. 세그멘테이션을 하려면 탐지 코드에서 무엇을 바꿔야 하나요?
3. `model.track()`에서 `persist=True`를 빼면 어떤 일이 생길까요?
4. `conf=0.5`를 `0.2`로 바꾸면 결과가 어떻게 달라질까요?
5. `deque(maxlen=30)` 대신 일반 리스트를 쓰면 어떤 문제가 생길까요?
6. `.gitignore`에 `*.pt`를 넣은 이유는 무엇인가요?

<details>
<summary>정답 보기</summary>

1. `py --list`
2. 모델 파일을 `yolo26n.pt` → `yolo26n-seg.pt` 로 바꾼다.
3. 프레임마다 추적 정보가 초기화되어, 같은 객체여도 ID가 유지되지 않는다.
4. 더 많은 객체를 찾지만, 잘못 찾은 객체(오탐)도 늘어난다.
5. 점이 끝없이 쌓여 메모리를 계속 쓰고, 꼬리가 화면 전체에 길게 남는다.
6. 모델 파일은 용량이 크고, 코드를 실행하면 자동으로 다시 받을 수 있기 때문이다.

</details>

### 도전 과제

- [ ] `MODEL_PATH`를 `yolo26s.pt`로 바꿔서 FPS와 정확도 차이를 비교해 보기
- [ ] `TRACKER`를 `botsort.yaml`로 바꿔서 ID가 얼마나 잘 유지되는지 비교해 보기
- [ ] `track.py`에서 `person`만 추적하도록 바꿔 보기 (힌트: `predict`/`track`의 `classes=[0]` 옵션)
- [ ] 화면에 가로선을 긋고, 선을 넘어간 사람 수를 세어 보기
