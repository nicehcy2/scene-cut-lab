# scene-cut-lab

브이로그/숏폼 영상에서 하이라이트 장면을 자동 추출하는 백엔드 파이프라인 프로토타입.

---

## 파이프라인

```
영상 입력
  → [1] 장면 분할 (PySceneDetect)
  → [2] STT 전사 (faster-whisper)
  → [3] Gemini 2.5 Flash 스코어링 (시각 + 발화 정보)
  → [4] 장면 선택 → results.json + highlight.mp4
```

---

## 설치

```bash
pip install -r requirements.txt
```

`.env.example`을 복사해 Gemini API 키를 입력한다.

```bash
cp .env.example .env
# GEMINI_API_KEY=your_api_key_here
```

---

## 실행

`inputs/` 폴더에 영상을 넣고 파일명만 입력하면 된다.

```bash
# 기본 실행
uv run python main.py input.mp4

# STT 모델 변경 (기본: small / 한국어 권장: medium)
uv run python main.py input.mp4 --stt-model medium

# STT 없이 시각 정보만으로 평가
uv run python main.py input.mp4 --no-stt

# 피사체 지정 (기본: 자동 감지)
uv run python main.py input.mp4 --subject 사람

# 점수 가중치 스타일 (기본: 장면 중심)
uv run python main.py input.mp4 --style 음성 중심

# 상위 N개만 추출
uv run python main.py input.mp4 --top-n 5

# maybe 장면도 포함 (기본: keep만)
uv run python main.py input.mp4 --include-maybe

# 장면 감지만 실행 (Gemini 없음, 분할 결과 확인용)
uv run python main.py input.mp4 --detect-only

# 기존 results.json으로 재선택/재렌더링
uv run python main.py input.mp4 --from-scores runs/.../results.json
```

---

## 출력

`runs/{영상명}_{시각}/results.json`

```json
{
  "scene": 3,
  "start": "00:00:42.500",
  "end": "00:01:05.200",
  "detected_subject": "사람",
  "quality_score": 85.0,
  "visual_score": 90.0,
  "speech_score": 70,
  "final_score": 82.0,
  "decision": "keep",
  "reason": "크게 웃으며 감정이 최고조에 달한 순간",
  "recommended_use": "메인 컷"
}
```

`decision`: `keep` (≥70) / `maybe` (≥50) / `drop`

---

## 주요 설정 (`config.py`)

| 항목 | 기본값 | 설명 |
|------|--------|------|
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini 모델 |
| `SCENE_THRESHOLD` | `15.0` | 장면 감지 민감도 (낮을수록 세분화) |
| `MAYBE_MIN_SCORE` | `64` | maybe 포함 최소 점수 (0~100) |
| `FRAMES_PER_SCENE` | `3` | 장면당 추출 프레임 수 |
