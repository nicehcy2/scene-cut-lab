"""
Gemini 2.5 Flash 기반 장면 하이라이트 스코어러
마스터 프롬프트 + 피사체 프롬프트를 조합해 점수를 산출한다.
"""
import io
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont

import config
from pipeline.frame_filter import filter_frames
from pipeline.scene_detector import Scene

MAX_CONCURRENT = 5  # 동시 Gemini API 요청 수 상한


def score_scenes(
    scenes: List[Scene],
    master_prompt: str,
    subject_prompts: dict,
    model_name: str,
    subject: Optional[str] = None,
) -> List[dict]:
    """각 장면의 프레임을 Gemini에 전송해 하이라이트 점수를 계산한다.

    subject가 지정되면 해당 피사체 프롬프트만 사용하고,
    None이면 모든 피사체 프롬프트를 포함해 Gemini가 자동 판단한다.
    """
    client = _build_client()
    semaphore = threading.Semaphore(MAX_CONCURRENT)
    print_lock = threading.Lock()

    # 4: subject_section을 루프 밖에서 한 번만 조립
    if subject and subject in subject_prompts:
        subject_section = subject_prompts[subject]
    else:
        subject_parts = []
        for name, prompt in subject_prompts.items():
            if name == "unknown":
                continue
            subject_parts.append(f"[{name} 기준]\n{prompt}")

        subject_section = "\n\n---\n\n".join(subject_parts)
        subject_section += f"\n\n---\n\n[unknown 기준]\n{subject_prompts.get('unknown', '')}"

    def process_scene(scene: Scene) -> tuple:
        valid_paths = filter_frames(
            scene.frame_paths,
            blur_threshold=config.BLUR_THRESHOLD,
            dark_threshold=config.DARK_THRESHOLD,
        )

        if not valid_paths:
            with print_lock:
                print(f"  [경고] Scene {scene.index}: 유효한 프레임 없음, DROP 처리")
            return scene.index, _make_fallback(scene)

        contents = _build_contents(valid_paths, master_prompt, subject_section, scene)
        with semaphore:  # 1: 동시 요청 수 제한
            response_text = _call_gemini(client, model_name, contents)

        parsed = _parse_response(response_text)
        result = _make_result(scene, parsed)

        decision_mark = {"keep": "✓", "maybe": "△", "drop": "✗"}.get(result["decision"], "?")
        with print_lock:
            print(
                f"  Scene {scene.index:3d} | {scene.start} ~ {scene.end} "
                f"| {decision_mark} {result['decision'].upper():5s} "
                f"| final={result['final_score']:.1f} "
                f"| {result['detected_subject']}"
                f"| {result['reason']}"
            )
        return scene.index, result

    # 1: 병렬 실행 후 원래 순서로 복원
    results_map: dict = {}
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        futures = {executor.submit(process_scene, scene): scene for scene in scenes}
        for future in as_completed(futures):
            idx, result = future.result()
            results_map[idx] = result

    return [results_map[scene.index] for scene in scenes]


_MAX_SIDE = 1280   # 원본 이미지 최대 해상도 (배치 전송 시)


def _pick_representative_frame(scene: Scene) -> Optional[str]:
    """장면에서 대표 프레임 1장을 선택한다.

    품질 필터 통과한 첫 번째 프레임 반환.
    전부 필터링되면 frame_paths 가운데 인덱스 반환 (최소 1장 보장).
    frame_paths가 비어 있으면 None 반환.
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
    return scene.frame_paths[len(scene.frame_paths) // 2]


def _build_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY가 설정되지 않았습니다. "
            "환경변수 또는 프로젝트 루트의 .env 파일에 키를 설정하세요."
        )
    return genai.Client(api_key=api_key)


def _build_contents(
    paths: List[str],
    master_prompt: str,
    subject_section: str,
    scene: Scene,
) -> list:
    parts = []

    for path in paths:
        img = Image.open(path).convert("RGB")
        img.thumbnail((1024, 1024), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"))

    full_prompt = (
        f"{master_prompt}\n\n"
        f"---\n\n"
        f"{subject_section}\n\n"
        f"---\n\n"
        f"지금 분석할 장면 정보:\n"
        f"- scene {scene.index}, {scene.start} ~ {scene.end}\n"
        f"- 위 이미지 {len(paths)}장이 이 장면의 대표 프레임이다.\n\n"
        f"반드시 아래 JSON 형식으로만 응답해라. JSON 외 다른 텍스트는 포함하지 마라:\n"
        f'{{"detected_subject": "사람|동물|풍경/공간|음식/음료|이동수단|사물/활동|unknown", '
        f'"secondary_subject": "사람|동물|...|null", '
        f'"applied_criteria": "사람 피사체 기준|동물 피사체 기준|...", '
        f'"general_drop": true|false, '
        f'"general_drop_reason": "해당 DROP 조건 한 줄|null", '
        f'"quality_score": 1~5, '
        f'"visual_score": 1~5, '
        f'"reason": "이유 한 줄 (한국어)"}}'
    )

    parts.append(types.Part.from_text(text=full_prompt))
    return [types.Content(role="user", parts=parts)]


def _call_gemini(client: genai.Client, model_name: str, contents: list, max_retry: int = 5) -> str:
    delay = 2.0
    for attempt in range(max_retry):
        try:
            # 2: JSON 응답 강제 → 파싱 오류 최소화
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            u = response.usage_metadata
            print(f"  [토큰] input={u.prompt_token_count}, output={u.candidates_token_count}, thinking={u.thoughts_token_count}")
            return response.text or ""
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
                # 서버 과부하 — jitter 추가해서 스레드들이 동시에 재시도하지 않도록
                wait = delay + random.uniform(0, 5)
                delay *= 2
                print(f"  [재시도 {attempt + 1}/{max_retry}] 서버 과부하 — {wait:.1f}초 대기 중...")
            else:
                wait = delay
                delay *= 2
                print(f"  [재시도 {attempt + 1}/{max_retry}] {e}")

            time.sleep(wait)
    return ""


def _parse_response(text: str) -> dict:
    try:
        data = json.loads(text.strip())
        if isinstance(data, list):
            data = data[0] if data else {}
        return _validate(data)
    except (json.JSONDecodeError, ValueError, KeyError, IndexError, AttributeError):
        return {
            "detected_subject": "unknown",
            "secondary_subject": None,
            "applied_criteria": "",
            "general_drop": False,
            "general_drop_reason": None,
            "quality_score": 3.0,
            "visual_score": 2.0,
            "reason": f"응답 파싱 실패: {text[:60]}",
        }


def _validate(data: dict) -> dict:
    def clamp(val, lo, hi, default):
        try:
            return min(hi, max(lo, float(val)))
        except (TypeError, ValueError):
            return default

    detected_subject = str(data.get("detected_subject", "unknown"))
    if detected_subject not in ("사람", "동물", "풍경/공간", "음식/음료", "이동수단", "사물/활동", "unknown"):
        detected_subject = "unknown"

    secondary = data.get("secondary_subject")
    if secondary is not None:
        secondary = str(secondary)
        if secondary.lower() in ("null", "none", ""):
            secondary = None

    general_drop_reason = data.get("general_drop_reason")
    if general_drop_reason is not None:
        general_drop_reason = str(general_drop_reason)
        if general_drop_reason.lower() in ("null", "none", ""):
            general_drop_reason = None

    return {
        "detected_subject": detected_subject,
        "secondary_subject": secondary,
        "applied_criteria": str(data.get("applied_criteria", "")),
        "general_drop": bool(data.get("general_drop", False)),
        "general_drop_reason": general_drop_reason,
        "quality_score": clamp(data.get("quality_score", 3), 1, 5, 3.0),
        "visual_score": clamp(data.get("visual_score", 2), 1, 5, 2.0),
        "reason": str(data.get("reason", "")),
    }


def _compute_backend_scores(
    quality_score: float,
    visual_score: float,
    general_drop: bool,
    speech_score: float = 3.0,
    style: str = "장면 중심",
) -> dict:
    if general_drop or visual_score <= 1:
        return {
            "speech_score": int(speech_score),
            "subject_score": 1.0,
            "final_score": 0.0,
            "decision": "drop",
            "recommended_use": "제거",
        }

    weight_map = {
        "장면 중심": (0.7, 0.3),
        "음성 중심": (0.3, 0.7),
        "균형": (0.5, 0.5),
    }
    w_v, w_s = weight_map.get(style, (0.7, 0.3))
    subject_score = visual_score * w_v + speech_score * w_s
    final_raw = quality_score * 0.4 + subject_score * 0.6

    if final_raw >= 3.5:
        decision = "keep"
    elif final_raw >= 2.5:
        decision = "maybe"
    else:
        decision = "drop"

    if decision == "drop":
        recommended_use = "제거"
    elif decision == "keep" and visual_score >= 4:
        recommended_use = "메인 컷"
    else:
        recommended_use = "보조 컷"

    return {
        "speech_score": int(speech_score),
        "subject_score": round(subject_score, 1),
        "final_score": round(final_raw, 1),
        "decision": decision,
        "recommended_use": recommended_use,
    }


def _make_result(scene: Scene, parsed: dict) -> dict:
    backend = _compute_backend_scores(
        quality_score=parsed["quality_score"],
        visual_score=parsed["visual_score"],
        general_drop=parsed["general_drop"],
    )
    return {
        "scene": scene.index,
        "start": scene.start,
        "end": scene.end,
        "frame_paths": scene.frame_paths,
        **parsed,
        **backend,
    }


def _make_fallback(scene: Scene) -> dict:
    return {
        "scene": scene.index,
        "start": scene.start,
        "end": scene.end,
        "detected_subject": "unknown",
        "secondary_subject": None,
        "applied_criteria": "",
        "general_drop": True,
        "general_drop_reason": "유효한 프레임 없음",
        "quality_score": 0,
        "visual_score": 0,
        "reason": "유효한 프레임 없음",
        "speech_score": 1,
        "subject_score": 0.0,
        "final_score": 0.0,
        "decision": "drop",
        "recommended_use": "제거",
    }


# ──────────────────────────────────────────────
# 그리드 방식 (맥락 파악용)
# ──────────────────────────────────────────────

_THUMB_W = 480
_THUMB_H = 270
_LABEL_H = 32
_CELL_PAD = 4


def score_scenes_grid(
    scenes: List[Scene],
    master_prompt: str,
    subject_prompts: dict,
    model_name: str,
    subject: Optional[str] = None,
    chunk_size: int = 12,
    grids_dir: Optional[str] = None,  # 하위 호환용 (더 이상 사용 안 함)
) -> List[dict]:
    """배치 방식: chunk_size개 장면의 원본 이미지를 묶어 Gemini에 전송.
    장면당 대표 프레임 1장, 이미지-메타 인터리브 방식.
    API 호출 횟수 = ceil(장면 수 / chunk_size).
    """
    client = _build_client()

    if subject and subject in subject_prompts:
        subject_section = subject_prompts[subject]
    else:
        subject_parts = []
        for name, prompt in subject_prompts.items():
            if name == "unknown":
                continue
            subject_parts.append(f"[{name} 기준]\n{prompt}")

        subject_section = "\n\n---\n\n".join(subject_parts)
        subject_section += f"\n\n---\n\n[unknown 기준]\n{subject_prompts.get('unknown', '')}"

    results_map: dict = {}
    chunks = [scenes[i:i + chunk_size] for i in range(0, len(scenes), chunk_size)]
    fallback_parsed = {
        "detected_subject": "unknown", "secondary_subject": None,
        "applied_criteria": "", "general_drop": False,
        "general_drop_reason": None, "quality_score": 3.0,
        "visual_score": 2.0, "reason": "응답 파싱 실패",
    }

    for chunk_idx, chunk in enumerate(chunks):
        print(f"  [배치 {chunk_idx + 1}/{len(chunks)}] Scene {chunk[0].index}~{chunk[-1].index} 분석 중...")

        # 장면별 대표 프레임 1장 선택
        batch = []
        for scene in chunk:
            frame_path = _pick_representative_frame(scene)
            if frame_path is None:
                print(f"  [경고] Scene {scene.index}: 프레임 없음 → DROP")
                results_map[scene.index] = _make_fallback(scene)
            else:
                batch.append((scene, frame_path))

        if not batch:
            continue

        contents = _build_batch_contents(batch, master_prompt, subject_section)
        response_text = _call_gemini(client, model_name, contents)
        parsed_by_scene = _parse_grid_response(response_text, [s for s, _ in batch])

        for scene, _ in batch:
            parsed = parsed_by_scene.get(scene.index, fallback_parsed)
            result = _make_result(scene, parsed)
            results_map[scene.index] = result
            decision_mark = {"keep": "✓", "maybe": "△", "drop": "✗"}.get(result["decision"], "?")
            print(
                f"  Scene {scene.index:3d} | {scene.start} ~ {scene.end} "
                f"| {decision_mark} {result['decision'].upper():5s} "
                f"| final={result['final_score']:.1f} "
                f"| {result['detected_subject']}"
                f"| {result['reason']}"
            )

    return [results_map[scene.index] for scene in scenes]


def _build_batch_contents(
    batch: List[tuple],   # (scene, frame_path)
    master_prompt: str,
    subject_section: str,
) -> list:
    """N장면 배치 호출 컨텐츠 생성 (원본 이미지 인터리브 방식).

    구조:
      [도입 텍스트] [이미지1][Scene 1 메타] [이미지2][Scene 2 메타] ... [분석 지시]
    """
    parts = []

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

    for scene, frame_path in batch:
        img = Image.open(frame_path).convert("RGB")
        img.thumbnail((_MAX_SIDE, _MAX_SIDE), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"))

        duration = scene.end_sec - scene.start_sec
        meta = (
            f"[Scene {scene.index}] {scene.start} ~ {scene.end} | {duration:.1f}s\n"
        )
        parts.append(types.Part.from_text(text=meta))

    n = len(batch)
    instruction = (
        f"\n위 {n}개 장면 각각에 대해 STEP 1~3을 수행해라.\n"
        f"반드시 아래 JSON 배열로만 응답해라 (원소 수: 정확히 {n}개, JSON 외 텍스트 없음):\n"
        f'[{{"scene": N, '
        f'"detected_subject": "사람|동물|풍경/공간|음식/음료|이동수단|사물/활동|unknown", '
        f'"secondary_subject": "사람|동물|...|null", '
        f'"applied_criteria": "...", '
        f'"general_drop": true|false, '
        f'"general_drop_reason": "...|null", '
        f'"quality_score": 1~5, "visual_score": 1~5, '
        f'"reason": "이유 한 줄 (한국어)"}}, ...]'
    )
    parts.append(types.Part.from_text(text=instruction))

    return [types.Content(role="user", parts=parts)]


def _build_grid_image(scenes: List[Scene]) -> Image.Image:
    n_cols = max((len(s.frame_paths) for s in scenes), default=3)
    n_cols = max(n_cols, 1)

    row_w = _CELL_PAD + n_cols * (_THUMB_W + _CELL_PAD)
    row_h = _LABEL_H + _THUMB_H + _CELL_PAD
    canvas = Image.new("RGB", (row_w, _CELL_PAD + len(scenes) * row_h), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    font = _load_grid_font(13)

    for row_idx, scene in enumerate(scenes):
        y = _CELL_PAD + row_idx * row_h
        duration = scene.end_sec - scene.start_sec
        label = f"Scene {scene.index} | {scene.start} ~ {scene.end} | {duration:.1f}s"
        draw.rectangle([_CELL_PAD, y, row_w - _CELL_PAD, y + _LABEL_H - 2], fill=(48, 48, 48))
        draw.text((_CELL_PAD + 6, y + 8), label, font=font, fill=(220, 220, 220))

        for col_idx, path in enumerate(scene.frame_paths[:n_cols]):
            x = _CELL_PAD + col_idx * (_THUMB_W + _CELL_PAD)
            fy = y + _LABEL_H
            try:
                thumb = Image.open(path).convert("RGB")
                # TODO: 해상도가 다른 영상이 섞이면 셀 높이가 불균일해질 수 있음
                thumb.thumbnail((_THUMB_W, _THUMB_H), Image.LANCZOS)
                canvas.paste(thumb, (x, fy))
            except Exception:
                pass

    canvas.thumbnail((2048, 2048), Image.LANCZOS)
    return canvas


def _build_grid_contents(
    grid_img: Image.Image,
    master_prompt: str,
    subject_section: str,
    scenes: List[Scene],
) -> list:
    buf = io.BytesIO()
    grid_img.save(buf, format="JPEG", quality=85)

    scene_meta = "\n".join(
        f"- Scene {s.index}: {s.start} ~ {s.end} ({s.end_sec - s.start_sec:.1f}s)"
        for s in scenes
    )

    prompt = (
        f"{master_prompt}\n\n"
        f"---\n\n"
        f"{subject_section}\n\n"
        f"---\n\n"
        f"위 이미지는 영상에서 추출한 {len(scenes)}개 장면의 그리드다.\n"
        f"각 행이 하나의 장면이며, 행 상단 라벨에 Scene 번호가 표시되어 있다.\n"
        f"전체 흐름과 장면 간 맥락을 고려해 상대적으로 평가해라.\n\n"
        f"장면 목록:\n{scene_meta}\n\n"
        f"각 장면을 순서대로 분석해 아래 JSON 배열로 응답해라 (원소 수: 반드시 {len(scenes)}개):\n"
        f'[{{"scene": N, '
        f'"detected_subject": "사람|동물|풍경/공간|음식/음료|이동수단|사물/활동|unknown", '
        f'"secondary_subject": "사람|동물|...|null", '
        f'"applied_criteria": "...", '
        f'"general_drop": true|false, '
        f'"general_drop_reason": "...|null", '
        f'"quality_score": 1~5, "visual_score": 1~5, '
        f'"reason": "이유 한 줄 (한국어)"}}, ...]'
    )

    parts = [
        types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"),
        types.Part.from_text(text=prompt),
    ]
    return [types.Content(role="user", parts=parts)]


def _parse_grid_response(text: str, scenes: List[Scene]) -> dict:
    fallback_item = {
        "detected_subject": "unknown",
        "secondary_subject": None,
        "applied_criteria": "",
        "general_drop": False,
        "general_drop_reason": None,
        "quality_score": 3.0,
        "visual_score": 2.0,
        "reason": "응답 파싱 실패",
    }

    if not text:
        print(f"  [경고] 응답이 비어있음 — fallback 처리")
        return {s.index: fallback_item.copy() for s in scenes}

    try:
        data = json.loads(text.strip())
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            print(f"  [경고] 그리드 응답이 배열이 아님: {text[:200]}")
            return {s.index: fallback_item.copy() for s in scenes}

        parsed_by_scene = {}
        for item in data:
            scene_no = item.get("scene")
            try:
                parsed = _validate(item)
            except Exception:
                parsed = fallback_item.copy()
            if scene_no is not None:
                parsed_by_scene[int(scene_no)] = parsed

        if not parsed_by_scene:
            print(f"  [경고] 그리드 응답 항목에 'scene' 키 없음 — fallback 처리. 원본:\n{text[:300]}")
            return {s.index: fallback_item.copy() for s in scenes}

        return parsed_by_scene
    except Exception:
        print(f"  [경고] 그리드 응답 JSON 파싱 실패. 원본:\n{text[:500]}")
        return {s.index: fallback_item.copy() for s in scenes}


def _load_grid_font(size: int):
    candidates = [
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/cour.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()
