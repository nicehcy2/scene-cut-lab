"""
Gate A: 완전 무음 세그먼트 제거
Gate B: NG 발화 패턴 감지 후 제거
"""
from typing import List

from pipeline.stt import SpeechSegment

_NG_PATTERNS = [
    "다시 할게요", "다시 할께요", "다시 하겠습니다", "다시 하겠어요",
    "잠깐만요", "잠깐요", "잠깐만", "잠깐",
    "NG", "컷", "편집해 주세요", "이거 빼주세요", "이거 편집해",
    "목 가다듬", "에헴",
]


def filter_silence(segments: List[SpeechSegment]) -> List[SpeechSegment]:
    """Gate A: 빈 텍스트 세그먼트 제거"""
    return [s for s in segments if s.text.strip()]


def filter_ng(segments: List[SpeechSegment]) -> List[SpeechSegment]:
    """Gate B: NG 발화 패턴 포함 세그먼트 제거"""
    result = []
    for seg in segments:
        text = seg.text.strip()
        if not any(p in text for p in _NG_PATTERNS):
            result.append(seg)
    return result
