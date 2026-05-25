"""
Gemini 2.5 Flash 기반 장면 하이라이트 스코어러
마스터 프롬프트 + 피사체 프롬프트를 조합해 점수를 산출한다.
"""
import io
import json
import os
import re
import time
from typing import List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

import config
from pipeline.frame_filter import filter_frames
from pipeline.scene_detector import Scene


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

    results = []
    for scene in scenes:
        valid_paths = filter_frames(
            scene.frame_paths,
            blur_threshold=config.BLUR_THRESHOLD,
            dark_threshold=config.DARK_THRESHOLD,
        )

        if not valid_paths:
            print(f"  [경고] Scene {scene.index}: 유효한 프레임 없음, DROP 처리")
            results.append(_make_fallback(scene))
            continue

        contents = _build_contents(valid_paths, master_prompt, subject_prompts, subject, scene)
        response_text = _call_gemini(client, model_name, contents)
        parsed = _parse_response(response_text)
        result = _make_result(scene, parsed)
        results.append(result)

        decision_mark = {"keep": "✓", "maybe": "△", "drop": "✗"}.get(result["decision"], "?")
        print(
            f"  Scene {scene.index:3d} | {scene.start} ~ {scene.end} "
            f"| {decision_mark} {result['decision'].upper():5s} "
            f"| final={result['final_score']:.1f} "
            f"| {result['main_subject']} "
            f"| {result['reason']}"
        )

    return results


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
    subject_prompts: dict,
    subject: Optional[str],
    scene: Scene,
) -> list:
    parts = []

    # 이미지 파트
    for path in paths:
        img = Image.open(path).convert("RGB")
        img.thumbnail((1024, 1024), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"))

    # 피사체 프롬프트 조합
    if subject and subject in subject_prompts:
        subject_section = subject_prompts[subject]
    else:
        # 피사체 미지정 → 전체 포함 (Gemini가 STEP 2에서 자동 선택)
        subject_section = "\n\n---\n\n".join(
            f"[{name} 기준]\n{prompt}"
            for name, prompt in subject_prompts.items()
            if name != "unknown"
        )
        subject_section += f"\n\n---\n\n[unknown 기준]\n{subject_prompts.get('unknown', '')}"

    full_prompt = (
        f"{master_prompt}\n\n"
        f"---\n\n"
        f"{subject_section}\n\n"
        f"---\n\n"
        f"지금 분석할 장면 정보:\n"
        f"- scene {scene.index}, {scene.start} ~ {scene.end}\n"
        f"- 위 이미지 {len(paths)}장이 이 장면의 대표 프레임이다.\n\n"
        f"반드시 아래 JSON 형식으로만 응답해라. JSON 외 다른 텍스트는 포함하지 마라:\n"
        f'{{"decision": "keep|maybe|drop", '
        f'"main_subject": "사람|동물|풍경/공간|음식/음료|이동수단|사물/활동|unknown", '
        f'"quality_score": 1~5, '
        f'"subject_score": 1~10, '
        f'"edit_score": 1~5, '
        f'"final_score": 소수점1자리, '
        f'"reason": "이유 한 줄 (한국어)", '
        f'"recommended_use": "메인 컷|리액션 컷|전환 컷|보조 컷|제거"}}'
    )

    parts.append(types.Part.from_text(text=full_prompt))
    return [types.Content(role="user", parts=parts)]


def _call_gemini(client: genai.Client, model_name: str, contents: list, max_retry: int = 3) -> str:
    delay = 1.0
    for attempt in range(max_retry):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
            )
            return response.text
        except Exception as e:
            if attempt == max_retry - 1:
                print(f"  [오류] Gemini API 호출 실패: {e}")
                return ""
            print(f"  [재시도 {attempt + 1}/{max_retry}] {e}")
            time.sleep(delay)
            delay *= 2
    return ""


def _parse_response(text: str) -> dict:
    # 1단계: 직접 파싱
    try:
        data = json.loads(text.strip())
        return _validate(data)
    except (json.JSONDecodeError, ValueError, KeyError):
        pass

    # 2단계: 정규식으로 JSON 블록 추출
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return _validate(data)
        except (json.JSONDecodeError, ValueError, KeyError):
            pass

    # 3단계: fallback
    return {
        "decision": "maybe",
        "main_subject": "unknown",
        "quality_score": 3.0,
        "subject_score": 3.0,
        "edit_score": 2.0,
        "final_score": 2.7,
        "reason": f"응답 파싱 실패: {text[:60]}",
        "recommended_use": "보조 컷",
    }


def _validate(data: dict) -> dict:
    decision = str(data.get("decision", "maybe")).lower()
    if decision not in ("keep", "maybe", "drop"):
        decision = "maybe"

    def clamp(val, lo, hi):
        return min(hi, max(lo, float(val)))

    quality = clamp(data.get("quality_score", 3), 1, 5)
    subject = clamp(data.get("subject_score", 3), 1, 10)
    edit = clamp(data.get("edit_score", 2), 1, 5)

    # subject_score를 5점 척도로 정규화 (HTML 기준이 1~10점)
    subject_norm = subject / 2.0

    final = round(quality * 0.3 + subject_norm * 0.4 + edit * 0.3, 1)

    # final_score가 명시된 경우 우선 사용, 없으면 계산값 사용
    if "final_score" in data:
        final = round(clamp(data["final_score"], 0, 5), 1)

    return {
        "decision": decision,
        "main_subject": str(data.get("main_subject", "unknown")),
        "quality_score": quality,
        "subject_score": subject,
        "edit_score": edit,
        "final_score": final,
        "reason": str(data.get("reason", "")),
        "recommended_use": str(data.get("recommended_use", "보조 컷")),
    }


def _make_result(scene: Scene, parsed: dict) -> dict:
    return {
        "scene": scene.index,
        "start": scene.start,
        "end": scene.end,
        **parsed,
    }


def _make_fallback(scene: Scene) -> dict:
    return {
        "scene": scene.index,
        "start": scene.start,
        "end": scene.end,
        "decision": "drop",
        "main_subject": "unknown",
        "quality_score": 0,
        "subject_score": 0,
        "edit_score": 0,
        "final_score": 0.0,
        "reason": "유효한 프레임 없음",
        "recommended_use": "제거",
    }
