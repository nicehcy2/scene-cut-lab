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
        HSV 컬러 변화 기반으로 컷 전환 지점 자동 감지
        장면당 대표 프레임 N장 추출 (기본값: 3장)
    ↓
[2단계] 프레임 품질 필터 (OpenCV)
        블러 프레임 제거 (Laplacian 분산 기준)
        암부 프레임 제거 (평균 밝기 기준)
        ※ 전부 필터링되면 원본 그대로 유지 (최소 1장 보장)
    ↓
[3단계] Gemini 2.5 Flash 하이라이트 스코어링
        마스터 프롬프트 + 피사체별 세부 프롬프트 조합
        → quality_score (기술 품질 1~5점)
        → visual_score  (장면 중요도 1~5점)
        → detected_subject (사람/동물/음식/풍경 등 자동 감지)
        → general_drop (기술 결함/NG/무의미 정적 즉시 제거)
        → final_score = quality×0.4 + visual×0.6 (백엔드 계산)
        → decision: keep (≥3.5) / maybe (≥2.5) / drop
    ↓
[4단계] 상위 N개 선택
        keep 전체 포함
        maybe는 final_score ≥ 3.2인 것만 포함
        final_score 내림차순 → 상위 N개 → 시간순 재정렬
    ↓
JSON 출력 (scene, start, end, scores, decision, reason 등)
    +
[선택] 하이라이트 영상 출력 (--export 사용 시)
        FFmpeg libx264 재인코딩으로 프레임 단위 정확한 컷
        각 구간 개별 컷 → 무손실 병합 → highlight.mp4
```

**Gemini 호출 방식 2가지**

| 모드 | 방식 | 특징 |
|---|---|---|
| `grid` (기본) | 12개 장면을 그리드 이미지 1장으로 묶어 Gemini 1회 호출 | 장면 간 맥락 파악 가능, API 호출 수 적음 |
| `parallel` | 장면별 개별 호출 (최대 5개 동시) | 각 장면을 독립적으로 정밀 분석 |

---

## 실행 모드

**전체 자동 모드 (기본)**
```bash
python main.py input.mp4
python main.py input.mp4 --export          # 하이라이트 영상 파일도 생성
```

**2단계 분리 모드**
```bash
# 1단계: 장면 감지만 실행 → scenes.json 저장
python main.py input.mp4 --detect-only

# 2단계: scenes.json 기반으로 영상 합치기만 실행
python main.py input.mp4 --from-scores runs/.../scenes.json --export
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
├── inputs/                          # 분석할 영상 파일 저장 위치
├── runs/                            # 실행 결과 저장 (실행 시마다 타임스탬프 폴더 생성)
│   └── {영상명}_{시각}/
│       ├── frames/                  # 추출된 대표 프레임
│       ├── grids/                   # 그리드 이미지 (grid 모드 / --detect-only)
│       ├── scenes.json              # --detect-only 결과
│       └── results.json             # 최종 하이라이트 결과
└── pipeline/
    ├── scene_detector.py            # [1단계] 장면 분할 + 프레임 추출
    ├── frame_filter.py              # [2단계] 블러/암부 프레임 필터
    ├── gemini_scorer.py             # [3단계] Gemini 기반 하이라이트 스코어러
    ├── highlight_selector.py        # [4단계] 상위 N개 선택
    ├── video_exporter.py            # [선택] FFmpeg 영상 컷 편집 및 병합
    ├── grid_builder.py              # 그리드 이미지 생성 (수동 리뷰용)
    ├── meta_extractor.py            # 장면 메타데이터 계산 (블러/밝기/유사도)
    └── clip_scorer_backup.py        # 이전 CLIP 스코어러 백업 (비활성)
```

---

## 기술 스택

| 라이브러리 | 역할 |
|---|---|
| `PySceneDetect` | 장면 전환 감지 (HSV 컬러 변화 기반) |
| `OpenCV` | 프레임 품질 필터 (Laplacian 분산, 밝기 계산) |
| `Pillow` | 이미지 로딩, 리사이즈, 그리드 생성 |
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
python main.py input.mp4 --export

# 피사체 유형 지정 (기본: 자동 감지)
# 선택지: 사람, 동물, 풍경/공간, 음식/음료, 이동수단, 사물/활동
python main.py input.mp4 --subject 사람 --export

# 상위 3개만 추출
python main.py input.mp4 --top-n 3 --export

# keep 장면만 포함 (maybe 제외)
python main.py input.mp4 --keep-only

# maybe 포함 최소 점수 조정 (기본값: 3.2)
python main.py input.mp4 --maybe-min-score 3.5

# Gemini 호출 방식 변경 (기본: grid)
python main.py input.mp4 --mode parallel

# 장면당 프레임 수 조정 (기본값: 3)
python main.py input.mp4 --frames-per-scene 5

# Gemini 모델 변경
python main.py input.mp4 --model gemini-2.0-flash

# 1단계만 실행 (장면 감지 + 프레임 추출)
python main.py input.mp4 --detect-only

# 저장된 scenes.json 기반으로 영상 합치기
python main.py input.mp4 --from-scores runs/input_20260527_120000/scenes.json --export
```

> `inputs/` 폴더 안에 영상을 넣으면 파일명만 입력해도 된다.  
> 실행 결과는 `runs/{영상명}_{시각}/` 폴더에 자동 저장된다.

---

## 출력 형식

```json
[
  {
    "scene": 3,
    "start": "00:00:42.500",
    "end": "00:01:05.200",
    "detected_subject": "사람",
    "secondary_subject": null,
    "applied_criteria": "사람 피사체 기준",
    "general_drop": false,
    "general_drop_reason": null,
    "quality_score": 4,
    "visual_score": 5,
    "reason": "크게 웃으며 감정이 최고조에 달한 순간",
    "speech_score": 3,
    "subject_score": 3.5,
    "final_score": 3.7,
    "decision": "keep",
    "recommended_use": "메인 컷"
  }
]
```

| 필드 | 설명 |
|---|---|
| `scene` | 장면 번호 |
| `start` / `end` | 시작/종료 타임코드 (HH:MM:SS.mmm) |
| `detected_subject` | Gemini가 감지한 주인공 피사체 |
| `secondary_subject` | 보조 피사체 (없으면 null) |
| `general_drop` | 기술 결함/NG/무의미 정적 해당 여부 |
| `quality_score` | 기술 품질 점수 1~5 (초점/흔들림/밝기/구도) |
| `visual_score` | 장면 중요도 점수 1~5 (피사체별 기준 적용) |
| `reason` | Gemini 판단 이유 한 줄 (한국어) |
| `final_score` | 백엔드 종합 점수 (quality×0.4 + subject_score×0.6) |
| `decision` | `keep` / `maybe` / `drop` |
| `recommended_use` | `메인 컷` / `보조 컷` / `제거` |

---

## 설정 (`config.py`)

### Gemini 모델
```python
GEMINI_MODEL = "gemini-2.5-flash"
```

### 피사체 선택지
```python
SUBJECT_CHOICES = ["사람", "동물", "풍경/공간", "음식/음료", "이동수단", "사물/활동", "unknown"]
```

### 프레임 품질 필터 임계값
```python
BLUR_THRESHOLD = 100.0   # Laplacian 분산. 이 값 미만이면 블러로 판정
DARK_THRESHOLD = 30.0    # 평균 밝기 (0-255). 이 값 미만이면 암부로 판정
```

### 장면 선택 기준
```python
MAYBE_MIN_SCORE = 3.2    # maybe 장면 포함 최소 점수
TOP_N = 5                # 최종 반환할 하이라이트 장면 수
```

### 기타
```python
FRAMES_PER_SCENE = 3     # 장면당 추출 프레임 수
```

---

## 알려진 이슈

### 영상 재생 오류 (Windows 기본 플레이어)

**증상**  
`highlight.mp4`가 Windows 기본 플레이어(영화 및 TV)에서 열리지 않는다.

**해결**  
`pipeline/video_exporter.py` FFmpeg 명령에 `-pix_fmt yuv420p` 옵션이 적용되어 있다. (현재 코드에 반영 완료)

### 영상 이음새 반복 구간

**증상**  
하이라이트 영상에서 같은 장면이 1~2초 반복되는 것처럼 보인다.

**원인**  
`-c copy` 스트림 복사 모드는 키프레임 단위로만 정확하게 잘리기 때문에 이전 장면 끝부분이 다음 클립 앞에 중복으로 붙을 수 있다.

**해결**  
`libx264 -preset ultrafast` 재인코딩으로 전환해 프레임 단위 정확한 컷을 적용했다. (현재 코드에 반영 완료)

---

## 변경 이력

### 2026-05-27 — 그리드 이미지 개선
- 그리드 이미지 저장 기능 추가 (`grids/` 폴더)
- 프레임 비율 유지 리사이즈 적용

### 2026-05-25 — CLIP → Gemini 2.5 Flash 전환

| 항목 | 이전 | 이후 |
|---|---|---|
| 스코어러 | `clip_scorer.py` (CLIP + PyTorch) | `gemini_scorer.py` (Gemini 2.5 Flash) |
| 점수 계산 방식 | 텍스트-이미지 코사인 유사도 차 | LLM이 장면 전체를 보고 직접 판단 |
| 출력 | `score` 단일 값 | `quality_score`, `visual_score`, `final_score`, `decision`, `reason` |
| 프레임 필터 | 없음 | 블러/암부 자동 제거 추가 |
| 피사체 인식 | 없음 | 6종 피사체 자동 감지 + 피사체별 세부 기준 적용 |
| 의존성 | `torch`, `torchvision`, `CLIP` | `google-genai`, `python-dotenv` |

기존 CLIP 스코어러는 `pipeline/clip_scorer_backup.py`로 보존되어 있다.
