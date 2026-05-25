"""
CLIP Zero-shot으로 장면별 하이라이트 점수 계산
"""
import os
from typing import List

import torch
import clip
from PIL import Image

from pipeline.scene_detector import Scene


def _load_model(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CLIP] 모델 로드 중: {model_name} ({device})")
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    return model, preprocess, device


def _score_frame(
    image_tensor: torch.Tensor,
    highlight_tokens: torch.Tensor,
    boring_tokens: torch.Tensor,
    model,
    device: str,
) -> float:
    image_tensor = image_tensor.to(device)
    highlight_tokens = highlight_tokens.to(device)
    boring_tokens = boring_tokens.to(device)

    with torch.no_grad():
        h_logits, _ = model(image_tensor, highlight_tokens)
        b_logits, _ = model(image_tensor, boring_tokens)

    # 각 프롬프트와의 유사도 평균값 차이
    return h_logits.mean().item() - b_logits.mean().item()


def score_scenes(
    scenes: List[Scene],
    highlight_prompts: List[str],
    boring_prompts: List[str],
    model_name: str,
) -> List[dict]:
    model, preprocess, device = _load_model(model_name)

    highlight_tokens = clip.tokenize(highlight_prompts)
    boring_tokens = clip.tokenize(boring_prompts)

    results = []
    for scene in scenes:
        valid_paths = [p for p in scene.frame_paths if os.path.exists(p)]
        if not valid_paths:
            print(f"  [경고] Scene {scene.index}: 추출된 프레임 없음, 점수=0")
            avg_score = 0.0
        else:
            frame_scores = []
            for path in valid_paths:
                image = preprocess(Image.open(path)).unsqueeze(0)
                score = _score_frame(image, highlight_tokens, boring_tokens, model, device)
                frame_scores.append(score)
            avg_score = sum(frame_scores) / len(frame_scores)

        results.append({
            "scene": scene.index,
            "start": scene.start,
            "end": scene.end,
            "score": round(avg_score, 4),
        })
        print(f"  Scene {scene.index:3d} | {scene.start} ~ {scene.end} | score={avg_score:.4f}")

    return results
