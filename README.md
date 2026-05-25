# scene-cut-lab

영상에서 하이라이트 장면을 자동으로 감지하고 타임스탬프를 추출하는 Python 파이프라인.

UI 없이 기능 검증 목적으로 만든 백엔드 파이프라인이며,  
브이로그/숏폼 원터치 자동편집 기능의 **장면 기반 컷편집 파이프라인 프로토타입**이다.

---

## 파이프라인 구조

```
영상 입력 (mp4)
    ↓
[1단계] 장면 분할 (PySceneDetect)
        컷 전환 지점을 자동 감지해 장면 단위로 분리
    ↓
[2단계] 대표 프레임 추출 (장면당 3장)
    ↓
[3단계] 프레임 품질 필터
        블러 / 암부 프레임 제거 (Laplacian 분산 + 평균 밝기 기준)
    ↓
[4단계] Gemini 2.5 Flash 하이라이트 스코어링
        장면당 1회 API 호출, 프레임 전체를 한 번에 전송
        → 점수 (0.0 ~ 1.0) + 판단 이유 반환
    ↓
[5단계] 상위 N개 선택
        점수 내림차순 → 상위 N개 → 시간순 재정렬
    ↓
JSON 출력 (scene, start, end, score, reason)
    +
[선택] 하이라이트 영상 출력 (--export 플래그 사용 시)
        FFmpeg으로 구간 컷 → 무손실 합치기 → highlight.mp4
```

---

## 파일 구조

```
scene-cut-lab/
├── main.py                          # CLI 엔트리포인트
├── config.py                        # 전체 설정값 (프롬프트, 임계값 등)
├── requirements.txt
├── .env.example                     # API 키 템플릿
├── .env                             # 실제 API 키 (gitignore 처리됨)
└── pipeline/
    ├── scene_detector.py            # 장면 분할 + 프레임 추출
    ├── frame_filter.py              # 블러/암부 프레임 필터
    ├── gemini_scorer.py             # Gemini 기반 하이라이트 스코어러
    ├── highlight_selector.py        # 상위 N개 선택
    └── clip_scorer_backup.py        # 이전 CLIP 스코어러 백업 (비활성)
```

---

## 기술 스택

| 라이브러리 | 역할 |
|---|---|
| `PySceneDetect` | 장면 전환 감지 (HSV 컬러 변화 기반) |
| `OpenCV` | 프레임 품질 필터 (Laplacian 분산, 밝기 계산) |
| `Pillow` | 이미지 로딩 및 리사이즈 |
| `google-genai` | Gemini 2.5 Flash API 호출 |
| `python-dotenv` | `.env` 파일에서 API 키 로드 |
| `imageio-ffmpeg` | FFmpeg 바이너리 제공 (영상 컷 편집 및 합치기) |

---

## 설치

```bash
pip install -r requirements.txt
```

---

## API 키 설정

`.env.example`을 복사해 `.env` 파일을 만들고 Gemini API 키를 입력한다.

```bash
cp .env.example .env
```

```
# .env
GEMINI_API_KEY=your_api_key_here
```

환경변수로 직접 설정해도 된다.

```bash
export GEMINI_API_KEY=your_api_key_here   # macOS/Linux
$env:GEMINI_API_KEY="your_api_key_here"   # Windows PowerShell
```

---

## 실행

```bash
# 기본 실행 — 타임스탬프 JSON만 출력
python main.py input.mp4

# 하이라이트 영상 파일도 함께 생성
python main.py input.mp4 --export highlight.mp4

# 영상 출력 + JSON 파일 저장 동시에
python main.py input.mp4 --export highlight.mp4 --output results.json

# 상위 3개만 추출
python main.py input.mp4 --top-n 3 --export highlight.mp4

# 장면당 프레임 수 조정 (기본값: 3)
python main.py input.mp4 --frames-per-scene 5

# Gemini 모델 변경
python main.py input.mp4 --model gemini-2.0-flash
```

---

## 출력 형식

```json
[
  {
    "scene": 3,
    "start": "00:00:42.500",
    "end": "00:01:05.200",
    "score": 0.87,
    "reason": "Friends laughing together at a scenic outdoor location."
  },
  {
    "scene": 7,
    "start": "00:02:11.000",
    "end": "00:02:34.800",
    "score": 0.81,
    "reason": "Exciting moment with dynamic movement and people enjoying food."
  }
]
```

| 필드 | 설명 |
|---|---|
| `scene` | 장면 번호 |
| `start` | 시작 타임코드 (HH:MM:SS.mmm) |
| `end` | 종료 타임코드 (HH:MM:SS.mmm) |
| `score` | 하이라이트 점수 (0.0 ~ 1.0, 높을수록 좋은 장면) |
| `reason` | Gemini가 판단한 이유 (영어 한 문장) |

---

## 설정 (`config.py`)

### Gemini 모델

```python
GEMINI_MODEL = "gemini-2.5-flash"
```

### 하이라이트 프롬프트

Gemini에게 "좋은 장면"의 기준으로 전달하는 텍스트 리스트.  
영상 도메인에 맞게 자유롭게 수정할 수 있다.

```python
HIGHLIGHT_PROMPTS = [
    "people laughing and having fun together",
    "exciting travel or outdoor moment",
    ...
]
```

### 제외 프롬프트

"나쁜 장면"의 기준.

```python
BORING_PROMPTS = [
    "boring empty scene with no people",
    "dark or blurry footage",
    ...
]
```

### 프레임 품질 필터 임계값

```python
BLUR_THRESHOLD = 100.0   # Laplacian 분산. 이 값 미만이면 블러로 판정
DARK_THRESHOLD = 30.0    # 평균 밝기 (0-255). 이 값 미만이면 암부로 판정
```

값을 올리면 더 엄격하게 필터링하고, 낮추면 더 많은 프레임을 통과시킨다.

### 기타

```python
FRAMES_PER_SCENE = 3     # 장면당 추출 프레임 수
TOP_N = 5                # 최종 반환할 하이라이트 장면 수
FRAMES_DIR = "frames"    # 추출된 프레임 저장 경로
```

---

## Gemini 스코어링 동작 방식

`pipeline/gemini_scorer.py`가 각 장면에 대해 다음을 수행한다.

1. 프레임 품질 필터를 거친 유효 프레임을 최대 1024px로 리사이즈
2. 모든 프레임 이미지 + 텍스트 프롬프트를 **1회 API 호출**로 Gemini에 전송
3. Gemini가 장면 전체를 보고 `{"score": 0.0~1.0, "reason": "..."}` JSON을 반환
4. JSON 파싱 실패 시 3단계 fallback 처리:
   - 직접 `json.loads` 시도
   - 정규식으로 JSON 블록 추출
   - 위 두 방법 모두 실패하면 기본값 `score=0.5` 반환
5. API 오류 시 최대 3회 재시도 (1초 → 2초 → 4초 지수 백오프)

---

## 알려진 이슈 및 해결 방법

### 영상 재생 오류 (Windows 영화 및 TV 앱)

**증상**  
`highlight.mp4` 가 VSCode 내장 플레이어에서는 재생되지만 Windows 기본 플레이어(영화 및 TV)에서 열리지 않는다.

**원인**  
`-c copy` 방식은 원본 픽셀 포맷을 그대로 유지하는데, 재인코딩(`libx264 -preset ultrafast`)으로 전환하면서 출력 픽셀 포맷이 Windows 기본 플레이어가 지원하지 않는 포맷으로 바뀔 수 있다.

**해결**  
`pipeline/video_exporter.py`의 FFmpeg 명령에 `-pix_fmt yuv420p` 옵션을 추가해 Windows 호환 픽셀 포맷으로 고정한다. (현재 코드에 적용 완료)

---

### `-c copy` 사용 시 영상 이음새 반복 구간

**증상**  
하이라이트 영상에서 같은 장면이 1~2초 반복되는 것처럼 보인다.

**원인**  
`-c copy`(스트림 복사) 모드는 키프레임 단위로만 정확하게 자를 수 있다. 장면 경계가 키프레임과 일치하지 않으면 직전 키프레임부터 잘리기 때문에 이전 장면 끝부분이 다음 클립 앞에 중복으로 붙는다.

**해결**  
`libx264 -preset ultrafast` 재인코딩으로 전환해 프레임 단위 정확한 컷을 적용했다. (현재 코드에 적용 완료)

---

## 변경 이력

### 2026-05-25 — CLIP → Gemini 2.5 Flash 전환

**배경**  
CLIP Zero-shot 방식은 텍스트-이미지 코사인 유사도만 계산해 영상 맥락을 이해하지 못하고 점수가 불안정했다.  
회사 Gemini API 사용 승인 후 멀티모달 LLM으로 교체했다.

**변경 내용**

| 항목 | 이전 | 이후 |
|---|---|---|
| 스코어러 | `clip_scorer.py` (CLIP + PyTorch) | `gemini_scorer.py` (Gemini 2.5 Flash) |
| 점수 계산 방식 | 텍스트-이미지 코사인 유사도 차 | LLM이 장면 전체를 보고 직접 판단 |
| 출력 | `score` 만 반환 | `score` + `reason` 반환 |
| 프레임 필터 | 없음 | 블러/암부 자동 제거 추가 |
| 의존성 | `torch`, `torchvision`, `CLIP` | `google-genai`, `python-dotenv` |
| 프롬프트 | 특정인(안경 쓴 남성) 기준 | 브이로그 범용 기준 |

**백업**  
기존 CLIP 스코어러는 `pipeline/clip_scorer_backup.py`로 보존되어 있다.
