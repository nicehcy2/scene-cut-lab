"""
Gemini 기반 장면 스코어러 V2 (STT 연동)

기존 gemini_scorer.py와의 차이점:
  - 장면당 대표 프레임 1장 (원본 이미지, 그리드 아님)
  - N장면씩 묶어서 Gemini 1회 호출 (기본 12장면/호출)
  - STT 스크립트를 프롬프트에 포함해 음성 맥락 제공
  - 점수 체계: 0~100점 (기존 1~5점에서 변경)
    - keep ≥ 70 / maybe ≥ 50 / drop < 50

highlight_selector.select_top() 호환:
  0~100 스케일이므로 maybe_min_score=64 로 호출할 것.
  예: select_top(results, top_n=5, maybe_min_score=64)

사용 예시:
    from pipeline.stt import transcribe
    from pipeline.scene_detector import detect_scenes
    from pipeline.gemini_scorer_v2 import score_scenes_v2

    stt_segments = transcribe("inputs/video.mp4")
    scenes = detect_scenes("inputs/video.mp4", frames_per_scene=1, frames_dir="runs/frames")
    results = score_scenes_v2(scenes, stt_segments)
"""
import os
import sys

# 직접 실행 시 프로젝트 루트를 Python 경로에 추가
# (python pipeline/gemini_scorer_v2.py 로 실행할 때 config, pipeline.* 인식)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import json
import random
import re
import time
from typing import List, Optional, Tuple

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

import config
from pipeline.frame_filter import filter_frames
from pipeline.scene_detector import Scene
from pipeline.stt import SpeechSegment, get_scene_transcript


# ── 점수 임계값 (0~100 스케일) ────────────────────────────────────────
SCORE_KEEP       = 70   # final_score ≥ 70 → keep
SCORE_MAYBE      = 50   # final_score ≥ 50 → maybe
SCORE_MAYBE_MIN  = 64   # highlight_selector.select_top() maybe_min_score 권장값

# 이미지 최대 해상도 (N장 배치이므로 단일 장면 호출보다 약간 축소)
_MAX_SIDE = 1280

# 배치 크기 기본값 (Gemini 1회 호출 당 장면 수)
DEFAULT_CHUNK_SIZE = 12

# 발화 존재 여부에 따른 speech_score 기본값
_SPEECH_SCORE_HAS    = 70.0   # 발화 있음
_SPEECH_SCORE_SILENT = 30.0   # 무음


# ── 프롬프트 ──────────────────────────────────────────────────────────

_SCALE_NOTE = (
    "\n\n※ 점수 스케일은 0~100점이다. 기존 1~5점 기준을 아래와 같이 매핑해라:\n"
    "- 5점 수준 → 90~100\n"
    "- 4점 수준 → 70~85\n"
    "- 3점 수준 → 50~65\n"
    "- 2점 수준 → 30~45\n"
    "- 1점 수준 (즉시 DROP) → 0~20\n"
)

# config.MASTER_PROMPT 기반으로 0~100 스케일로 수정
_MASTER_PROMPT_V2 = (
    config.MASTER_PROMPT
    .replace(
        "quality_score (1~5점): 기술 품질 (초점/흔들림/밝기/구도)",
        "quality_score (0~100점): 기술 품질 (초점/흔들림/밝기/구도)",
    )
    .replace(
        "visual_score (1~5점): 장면 중요도 — 각 피사체 프롬프트 기준 적용",
        "visual_score (0~100점): 장면 중요도 — 각 피사체 프롬프트 기준 적용",
    )
    .replace(
        "1점: 기술적으로 매우 나쁨\n2점: 사용하기 아쉬운 수준\n3점: 보통. 사용 가능한 수준\n4점: 안정적인 기술 품질\n5점: 초점/구도/밝기 모두 완벽",
        "0~20: 기술적으로 매우 나쁨\n30~45: 사용하기 아쉬운 수준\n50~65: 보통. 사용 가능한 수준\n70~85: 안정적인 기술 품질\n90~100: 초점/구도/밝기 모두 완벽",
    )
)


# ── 공개 API ──────────────────────────────────────────────────────────

def score_scenes_v2(
    scenes: List[Scene],
    stt_segments: List[SpeechSegment],
    model_name: str = config.GEMINI_MODEL,
    subject: Optional[str] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    style: str = "장면 중심",
) -> List[dict]:
    """V2 스코어러 메인 함수.

    Args:
        scenes:       detect_scenes() 결과
        stt_segments: transcribe() 결과. STT 없이 실행하려면 [] 전달
        model_name:   Gemini 모델명
        subject:      피사체 고정 지정 (None 이면 자동 감지)
        chunk_size:   한 번에 Gemini에 보낼 장면 수 (기본 12)
        style:        점수 가중치 스타일
                      "장면 중심" (visual 0.7 / speech 0.3) |
                      "음성 중심" (visual 0.3 / speech 0.7) |
                      "균형"     (visual 0.5 / speech 0.5)

    Returns:
        List[dict] — highlight_selector.select_top()에 바로 전달 가능.
        ※ 0~100 스케일이므로 maybe_min_score=SCORE_MAYBE_MIN(64) 으로 호출할 것.
    """
    client = _build_client()
    subject_section = _build_subject_section(subject, config.SUBJECT_PROMPTS)

    results_map: dict = {}
    chunks = [scenes[i:i + chunk_size] for i in range(0, len(scenes), chunk_size)]

    for chunk_idx, chunk in enumerate(chunks):
        print(
            f"\n[V2 스코어러] 배치 {chunk_idx + 1}/{len(chunks)} "
            f"— Scene {chunk[0].index}~{chunk[-1].index} ({len(chunk)}개)"
        )

        # ① 장면별 대표 프레임 + STT 스크립트 준비
        batch: List[Tuple[Scene, Optional[str], str]] = []
        for scene in chunk:
            frame_path = _pick_representative_frame(scene)
            transcript = get_scene_transcript(stt_segments, scene.start_sec, scene.end_sec)
            batch.append((scene, frame_path, transcript))

        # ② 프레임 없는 장면 → fallback 처리, 나머지만 API 호출
        valid_batch = []
        for scene, frame_path, transcript in batch:
            if frame_path is None:
                print(f"  [경고] Scene {scene.index}: 프레임 없음 → DROP")
                results_map[scene.index] = _make_fallback_v2(scene, "프레임 없음")
            else:
                valid_batch.append((scene, frame_path, transcript))

        if not valid_batch:
            continue

        # ③ Gemini API 호출
        contents = _build_batch_contents(valid_batch, _MASTER_PROMPT_V2, subject_section)
        response_text = _call_gemini(client, model_name, contents)

        # ④ 응답 파싱 → 결과 생성
        parsed_by_scene = _parse_batch_response(
            response_text, [s for s, _, _ in valid_batch]
        )

        for scene, frame_path, transcript in valid_batch:
            parsed = parsed_by_scene.get(scene.index)
            if parsed is None:
                parsed = _fallback_parsed_template()
                parsed["reason"] = "응답에서 해당 장면 누락"

            speech_score = _calc_speech_score(transcript)
            result = _make_result_v2(scene, parsed, speech_score, style)
            results_map[scene.index] = result

            mark = {"keep": "✓", "maybe": "△", "drop": "✗"}.get(result["decision"], "?")
            preview = f'"{transcript[:28]}…"' if transcript else "(무음)"
            print(
                f"  Scene {scene.index:3d} | {scene.start} ~ {scene.end}"
                f" | {mark} {result['decision'].upper():5s}"
                f" | Q={result['quality_score']:.0f}"
                f" V={result['visual_score']:.0f}"
                f" S={result['speech_score']}"
                f" → {result['final_score']:.1f}"
                f" | {result['detected_subject']}"
                f" | {preview}"
            )

    # 원래 순서 유지, API 호출 누락된 장면은 fallback
    output = []
    for scene in scenes:
        if scene.index in results_map:
            output.append(results_map[scene.index])
        else:
            output.append(_make_fallback_v2(scene, "처리 누락"))
    return output


# ── 내부 유틸리티 ─────────────────────────────────────────────────────

def _build_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY가 설정되지 않았습니다. "
            "환경변수 또는 .env 파일에 키를 설정하세요."
        )
    return genai.Client(api_key=api_key)


def _pick_representative_frame(scene: Scene) -> Optional[str]:
    """장면에서 대표 프레임 1장을 선택한다.

    품질 필터를 통과한 첫 번째 프레임을 반환한다.
    전부 필터링되면 frame_paths 가운데 인덱스를 반환한다 (최소 1장 보장).
    frame_paths가 비어 있으면 None을 반환한다.
    """
    if not scene.frame_paths:
        return None

    valid = filter_frames(
        scene.frame_paths,
        blur_threshold=config.BLUR_THRESHOLD,
        dark_threshold=config.DARK_THRESHOLD,
    )
    if valid:
        return valid[0]
    # 전부 필터링되어도 가운데 프레임은 반환 (fallback)
    return scene.frame_paths[len(scene.frame_paths) // 2]


def _calc_speech_score(transcript: str) -> float:
    """STT 결과 기반 speech_score 계산 (0~100).

    현재: 발화 존재 여부만 반영하는 간단한 구현.
    TODO: 발화 밀도, 단어 신뢰도, 키워드 등으로 고도화 가능.
      - 발화 있음: 70
      - 무음:      30
    """
    if transcript and transcript.strip():
        return _SPEECH_SCORE_HAS
    return _SPEECH_SCORE_SILENT


def _build_subject_section(subject: Optional[str], subject_prompts: dict) -> str:
    """피사체 기준 섹션 생성 (0~100 스케일 매핑 주석 포함)."""
    if subject and subject in subject_prompts:
        return subject_prompts[subject] + _SCALE_NOTE

    parts = []
    for name, prompt in subject_prompts.items():
        if name == "unknown":
            continue
        parts.append(f"[{name} 기준]\n{prompt}")

    section  = "\n\n---\n\n".join(parts)
    section += f"\n\n---\n\n[unknown 기준]\n{subject_prompts.get('unknown', '')}"
    section += _SCALE_NOTE
    return section


def _build_batch_contents(
    batch: List[Tuple[Scene, str, str]],   # (scene, frame_path, transcript)
    master_prompt: str,
    subject_section: str,
) -> list:
    """N장면 배치 호출 컨텐츠를 생성한다.

    구조:
      [도입 텍스트 (마스터 프롬프트 + 피사체 기준)]
      [이미지1] [Scene 1 메타]
      [이미지2] [Scene 2 메타]
      ...
      [분석 지시 + JSON 포맷]

    이미지와 메타 정보를 인터리브해 Gemini가 이미지-장면을 정확히 매핑하게 한다.
    """
    parts = []

    # 도입: 마스터 프롬프트 + 피사체 기준
    intro = (
        f"{master_prompt}\n\n"
        f"---\n\n"
        f"{subject_section}\n\n"
        f"---\n\n"
        f"아래 {len(batch)}개 장면을 순서대로 분석해라.\n"
        f"각 장면은 [이미지] → [장면 정보] 순으로 제공된다.\n"
        f"전체 흐름과 장면 간 맥락을 고려해 상대적으로 평가해라.\n\n"
    )
    parts.append(types.Part.from_text(text=intro))

    # 각 장면: 이미지 → 메타 텍스트 인터리브
    for scene, frame_path, transcript in batch:
        img = Image.open(frame_path).convert("RGB")
        img.thumbnail((_MAX_SIDE, _MAX_SIDE), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"))

        duration     = scene.end_sec - scene.start_sec
        speech_line  = f'"{transcript}"' if transcript and transcript.strip() else "(발화 없음)"
        meta = (
            f"[Scene {scene.index}] {scene.start} ~ {scene.end} | {duration:.1f}s\n"
            f"발화: {speech_line}\n"
        )
        parts.append(types.Part.from_text(text=meta))

    # 분석 지시 + JSON 포맷
    n = len(batch)
    instruction = (
        f"\n위 {n}개 장면 각각에 대해 STEP 1~3을 수행해라.\n"
        f"반드시 아래 JSON 배열로만 응답해라 (원소 수: 정확히 {n}개, JSON 외 텍스트 없음):\n"
        f'[{{"scene": N, '
        f'"detected_subject": "사람|동물|풍경/공간|음식/음료|이동수단|사물/활동|unknown", '
        f'"secondary_subject": "사람|동물|...|null", '
        f'"applied_criteria": "적용된 피사체 기준", '
        f'"general_drop": true|false, '
        f'"general_drop_reason": "이유 또는 null", '
        f'"quality_score": 0~100, '
        f'"visual_score": 0~100, '
        f'"reason": "이유 한 줄 (한국어)"}}, ...]'
    )
    parts.append(types.Part.from_text(text=instruction))

    return [types.Content(role="user", parts=parts)]


def _call_gemini(
    client: genai.Client,
    model_name: str,
    contents: list,
    max_retry: int = 5,
) -> str:
    """Gemini API 호출 (지수 백오프 재시도 포함)."""
    delay = 2.0
    for attempt in range(max_retry):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            u = response.usage_metadata
            print(
                f"  [토큰] input={u.prompt_token_count}, "
                f"output={u.candidates_token_count}, "
                f"thinking={u.thoughts_token_count}"
            )
            return response.text

        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "RESOURCE_EXHAUSTED" in err_str

            if attempt == max_retry - 1:
                print(f"  [오류] Gemini API 호출 실패: {e}")
                return ""

            if is_rate_limit:
                match = re.search(r"retryDelay.*?(\d+)s", err_str)
                wait = int(match.group(1)) + 2 if match else 60
                print(f"  [재시도 {attempt + 1}/{max_retry}] Rate Limit — {wait}초 대기 중...")
            elif "503" in err_str or "UNAVAILABLE" in err_str:
                wait = delay + random.uniform(0, 5)
                delay *= 2
                print(f"  [재시도 {attempt + 1}/{max_retry}] 서버 과부하 — {wait:.1f}초 대기 중...")
            else:
                wait = delay
                delay *= 2
                print(f"  [재시도 {attempt + 1}/{max_retry}] {e}")

            time.sleep(wait)
    return ""


def _parse_batch_response(text: str, scenes: List[Scene]) -> dict:
    """배치 응답 JSON 파싱 → {scene_index: parsed_dict} 반환."""
    fallback_item = _fallback_parsed_template()

    try:
        data = json.loads(text.strip())
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            print(f"  [경고] 배치 응답이 배열이 아님: {text[:200]}")
            return {s.index: fallback_item.copy() for s in scenes}

        parsed_by_scene = {}
        for item in data:
            scene_no = item.get("scene")
            try:
                parsed = _validate_v2(item)
            except Exception:
                parsed = fallback_item.copy()
            if scene_no is not None:
                parsed_by_scene[int(scene_no)] = parsed

        if not parsed_by_scene:
            print(f"  [경고] 응답 항목에 'scene' 키 없음 — fallback. 원본:\n{text[:300]}")
            return {s.index: fallback_item.copy() for s in scenes}

        return parsed_by_scene

    except Exception:
        print(f"  [경고] JSON 파싱 실패. 원본:\n{text[:500]}")
        return {s.index: fallback_item.copy() for s in scenes}


def _validate_v2(data: dict) -> dict:
    """응답 데이터 유효성 검증 및 정규화 (0~100 스케일)."""
    def clamp(val, lo, hi, default):
        try:
            return min(hi, max(lo, float(val)))
        except (TypeError, ValueError):
            return default

    detected_subject = str(data.get("detected_subject", "unknown"))
    if detected_subject not in (
        "사람", "동물", "풍경/공간", "음식/음료", "이동수단", "사물/활동", "unknown"
    ):
        detected_subject = "unknown"

    secondary = data.get("secondary_subject")
    if secondary is not None:
        secondary = str(secondary)
        if secondary.lower() in ("null", "none", ""):
            secondary = None

    drop_reason = data.get("general_drop_reason")
    if drop_reason is not None:
        drop_reason = str(drop_reason)
        if drop_reason.lower() in ("null", "none", ""):
            drop_reason = None

    return {
        "detected_subject":    detected_subject,
        "secondary_subject":   secondary,
        "applied_criteria":    str(data.get("applied_criteria", "")),
        "general_drop":        bool(data.get("general_drop", False)),
        "general_drop_reason": drop_reason,
        "quality_score":       clamp(data.get("quality_score", 50), 0, 100, 50.0),
        "visual_score":        clamp(data.get("visual_score", 40), 0, 100, 40.0),
        "reason":              str(data.get("reason", "")),
    }


def _fallback_parsed_template() -> dict:
    return {
        "detected_subject":    "unknown",
        "secondary_subject":   None,
        "applied_criteria":    "",
        "general_drop":        False,
        "general_drop_reason": None,
        "quality_score":       50.0,
        "visual_score":        40.0,
        "reason":              "응답 파싱 실패",
    }


def _compute_backend_scores_v2(
    quality_score: float,
    visual_score: float,
    general_drop: bool,
    speech_score: float,
    style: str = "장면 중심",
) -> dict:
    """백엔드 점수 계산 (0~100 스케일).

    final_score = quality × 0.4 + subject_score × 0.6
    subject_score = visual × w_v + speech × w_s

    가중치:
      장면 중심 — visual 0.7 / speech 0.3
      음성 중심 — visual 0.3 / speech 0.7
      균형     — visual 0.5 / speech 0.5

    임계값:
      keep  ≥ 70
      maybe ≥ 50
      drop  < 50
    """
    # general_drop 또는 visual_score ≤ 20 (1점 수준) → 즉시 DROP
    if general_drop or visual_score <= 20:
        return {
            "speech_score":    round(speech_score),
            "subject_score":   0.0,
            "final_score":     0.0,
            "decision":        "drop",
            "recommended_use": "제거",
        }

    weight_map = {
        "장면 중심": (0.7, 0.3),
        "음성 중심": (0.3, 0.7),
        "균형":      (0.5, 0.5),
    }
    w_v, w_s = weight_map.get(style, (0.7, 0.3))
    subject_score = visual_score * w_v + speech_score * w_s
    final_raw     = quality_score * 0.4 + subject_score * 0.6

    if final_raw >= SCORE_KEEP:
        decision = "keep"
    elif final_raw >= SCORE_MAYBE:
        decision = "maybe"
    else:
        decision = "drop"

    if decision == "drop":
        recommended_use = "제거"
    elif decision == "keep" and visual_score >= 70:
        recommended_use = "메인 컷"
    else:
        recommended_use = "보조 컷"

    return {
        "speech_score":    round(speech_score),
        "subject_score":   round(subject_score, 1),
        "final_score":     round(final_raw, 1),
        "decision":        decision,
        "recommended_use": recommended_use,
    }


def _make_result_v2(
    scene: Scene,
    parsed: dict,
    speech_score: float,
    style: str,
) -> dict:
    backend = _compute_backend_scores_v2(
        quality_score=parsed["quality_score"],
        visual_score=parsed["visual_score"],
        general_drop=parsed["general_drop"],
        speech_score=speech_score,
        style=style,
    )
    return {
        "scene": scene.index,
        "start": scene.start,
        "end":   scene.end,
        **parsed,
        **backend,
    }


def _make_fallback_v2(scene: Scene, reason: str) -> dict:
    return {
        "scene":               scene.index,
        "start":               scene.start,
        "end":                 scene.end,
        "detected_subject":    "unknown",
        "secondary_subject":   None,
        "applied_criteria":    "",
        "general_drop":        True,
        "general_drop_reason": reason,
        "quality_score":       0,
        "visual_score":        0,
        "reason":              reason,
        "speech_score":        0,
        "subject_score":       0.0,
        "final_score":         0.0,
        "decision":            "drop",
        "recommended_use":     "제거",
    }


# ── 단독 실행 (간단 테스트) ──────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import tempfile

    from pipeline.scene_detector import detect_scenes
    from pipeline.stt import transcribe

    if len(sys.argv) < 2:
        print("사용법: python pipeline/gemini_scorer_v2.py <영상경로> [--style 장면중심|음성중심|균형]")
        print("예시:   python pipeline/gemini_scorer_v2.py inputs/video.mp4")
        sys.exit(1)

    video = sys.argv[1]

    style_arg = "장면 중심"
    if "--style" in sys.argv:
        idx = sys.argv.index("--style")
        if idx + 1 < len(sys.argv):
            style_arg = sys.argv[idx + 1]

    print(f"[1/3] STT 전사 중: {video}")
    segs = transcribe(video)

    print(f"\n[2/3] 장면 분할 중 (대표 프레임 1장/장면)")
    frames_dir = os.path.join(tempfile.mkdtemp(), "frames_v2")
    scenes = detect_scenes(video, frames_per_scene=1, frames_dir=frames_dir)

    print(f"\n[3/3] V2 스코어링 중 (스타일: {style_arg})")
    results = score_scenes_v2(scenes, segs, style=style_arg)

    # 결과 출력
    print(f"\n{'─' * 80}")
    print(f"{'Scene':>5}  {'Start':>12}  {'End':>12}  {'Q':>4} {'V':>4} {'S':>4}  {'Final':>5}  Decision")
    print(f"{'─' * 80}")
    for r in results:
        mark = {"keep": "✓", "maybe": "△", "drop": "✗"}.get(r["decision"], "?")
        print(
            f"{r['scene']:>5}  {r['start']:>12}  {r['end']:>12}"
            f"  {r['quality_score']:>4.0f} {r['visual_score']:>4.0f} {r['speech_score']:>4}"
            f"  {r['final_score']:>5.1f}  {mark} {r['decision']}"
        )
    print(f"{'─' * 80}")

    keep  = sum(1 for r in results if r["decision"] == "keep")
    maybe = sum(1 for r in results if r["decision"] == "maybe")
    drop  = sum(1 for r in results if r["decision"] == "drop")
    print(f"keep: {keep}  maybe: {maybe}  drop: {drop}  합계: {len(results)}")
    print(f"\n※ highlight_selector.select_top() 호출 시 maybe_min_score={SCORE_MAYBE_MIN} 사용 권장")
