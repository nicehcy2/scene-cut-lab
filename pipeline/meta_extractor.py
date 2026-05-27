"""
장면별 메타데이터 사전 계산 (GPT 수동 채점 보조용)

계산 항목:
  blur_score         - Laplacian 분산 평균 (높을수록 선명)
  brightness         - 평균 밝기 0~255
  similarity_to_prev - 이전 장면 마지막 프레임과의 HSV 히스토그램 유사도 (0~1)
  is_blurry          - blur_score < BLUR_THRESHOLD
  is_dark            - brightness < DARK_THRESHOLD
"""
from typing import Dict, List, Optional

import cv2
import numpy as np

import config
from pipeline.scene_detector import Scene


def compute_meta(scenes: List[Scene]) -> List[Dict]:
    """각 장면의 메타데이터를 계산해 리스트로 반환. scenes와 순서 동일."""
    results = []
    prev_last_img: Optional[np.ndarray] = None

    for scene in scenes:
        blur_scores, brightness_scores = [], []
        last_img = None

        for path in scene.frame_paths:
            img = cv2.imread(path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur_scores.append(cv2.Laplacian(gray, cv2.CV_64F).var())
            brightness_scores.append(float(np.mean(gray)))
            last_img = img

        blur = round(float(np.mean(blur_scores)), 1) if blur_scores else 0.0
        brightness = round(float(np.mean(brightness_scores)), 1) if brightness_scores else 0.0

        # 이전 장면과 유사도
        similarity: Optional[float] = None
        if prev_last_img is not None and scene.frame_paths:
            first_img = cv2.imread(scene.frame_paths[0])
            if first_img is not None:
                similarity = round(_hist_similarity(prev_last_img, first_img), 3)

        results.append({
            "blur_score": blur,
            "brightness": brightness,
            "similarity_to_prev": similarity,
            "is_blurry": blur < config.BLUR_THRESHOLD,
            "is_dark": brightness < config.DARK_THRESHOLD,
        })

        if last_img is not None:
            prev_last_img = last_img

    return results


def _hist_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """HSV 히스토그램 코릴레이션 유사도 (0~1, 1에 가까울수록 유사)."""
    def hist(img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(h, h)
        return h

    return float(cv2.compareHist(hist(img1), hist(img2), cv2.HISTCMP_CORREL))
