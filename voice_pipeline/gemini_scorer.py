"""
음성 중심 파이프라인 — 텍스트 전용 Gemini 스코어러
이미지 없이 STT 세그먼트 텍스트만으로 importance 5/3/1 산출
"""
import json
import os
import time
from typing import List

from dotenv import load_dotenv
from google import genai

import config
from pipeline.stt import SpeechSegment

load_dotenv()

_PROMPT = """# 역할
너는 영상 편집 보조 AI다.
전체 스크립트를 읽고 핵심 발화 구간을 선별해라.
이미지는 없다. 텍스트만으로 판단해라.

# STEP 1 — 주제 파악
전체 스크립트를 읽고 이 영상의 핵심 주제를 1~2문장으로 정의해라.

# STEP 2 — DROP 조건 (해당하면 keep: false)
- 말 더듬, 반복 발화 (같은 내용 두 번 이상)
- 주제와 무관한 잡담·여담
- 메타 발화 ("잠깐요", "다시 할게요", "NG" 등)
- 중복 내용 중 전달력이 낮은 구간

# STEP 3 — 핵심 구간 선별 + importance 산출
주제를 기준으로 반드시 포함할 구간을 선별해라.
중복 내용은 더 명확하게 전달된 한 구간만 선택해라.

5: 영상 주제를 직접 설명하는 핵심 발화 — 반드시 포함
3: 주제와 간접 연관, 흐름상 필요한 구간
1: 주제와 무관 → DROP 대상

# 출력 형식 (JSON 배열만 출력, 다른 텍스트 없음)
[
  {
    "clip_id": "세그먼트 인덱스 (0부터)",
    "keep": true | false,
    "importance": 5 | 3 | 1,
    "reason": "한 줄 판단 요약"
  }
]"""


def _build_transcript(segments: List[SpeechSegment]) -> str:
    lines = [f"[{i}] {s.start:.1f}s~{s.end:.1f}s: {s.text}" for i, s in enumerate(segments)]
    return "\n".join(lines)


def _sec_to_tc(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def score_segments(
    segments: List[SpeechSegment],
    model_name: str = config.GEMINI_MODEL,
) -> List[dict]:
    """STT 세그먼트를 텍스트만으로 Gemini에 평가 요청, importance 기반 결과 반환.

    반환 형식은 highlight_selector.select_top()에 바로 전달 가능.
    """
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    transcript = _build_transcript(segments)
    prompt = f"{_PROMPT}\n\n# 전체 스크립트\n{transcript}"

    parsed = None
    for attempt in range(5):
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            text = response.text.strip()
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            parsed = json.loads(text.strip())
            break
        except Exception as e:
            if attempt == 4:
                raise
            wait = 2 ** attempt
            print(f"[Voice Gemini] 재시도 {attempt + 1}/5 ({wait}s): {e}")
            time.sleep(wait)

    results = []
    for item in parsed:
        idx = int(item["clip_id"])
        if idx >= len(segments):
            continue
        seg = segments[idx]
        importance = int(item.get("importance", 1))
        keep_flag = bool(item.get("keep", False)) and importance > 1

        if importance == 5 and keep_flag:
            decision, final_score = "keep", 100.0
        elif importance == 3 and keep_flag:
            decision, final_score = "maybe", 80.0
        else:
            decision, final_score = "drop", 20.0

        results.append({
            "start": _sec_to_tc(seg.start),
            "end": _sec_to_tc(seg.end),
            "start_sec": seg.start,
            "end_sec": seg.end,
            "text": seg.text,
            "importance": importance,
            "reason": item.get("reason", ""),
            "decision": decision,
            "final_score": final_score,
        })

    return results
