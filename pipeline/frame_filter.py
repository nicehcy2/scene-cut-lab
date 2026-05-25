"""
프레임 품질 필터 - blur/암부 프레임 제거
"""
import os
from typing import List

import cv2
import numpy as np


def filter_frames(
    frame_paths: List[str],
    blur_threshold: float = 100.0,
    dark_threshold: float = 30.0,
) -> List[str]:
    """블러/어두운 프레임을 제거하고 유효한 경로만 반환.

    모든 프레임이 필터링될 경우 원본 리스트를 그대로 반환해 최소 1장을 보장한다.
    """
    valid = []
    for path in frame_paths:
        if not os.path.exists(path):
            continue
        img = cv2.imread(path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if _laplacian_variance(gray) < blur_threshold:
            continue
        if _mean_brightness(gray) < dark_threshold:
            continue
        valid.append(path)

    return valid if valid else list(frame_paths)


def _laplacian_variance(gray: np.ndarray) -> float:
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _mean_brightness(gray: np.ndarray) -> float:
    return float(np.mean(gray))
