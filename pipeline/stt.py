"""
faster-whisper 기반 STT 모듈

영상 전체를 전사(transcribe)하고,
장면 시간 범위에 해당하는 스크립트를 슬라이싱해서 반환한다.

사용 예시:
    segments = transcribe("inputs/video.mp4")
    text = get_scene_transcript(segments, start_sec=10.0, end_sec=25.5)
"""
import time
from dataclasses import dataclass
from typing import List, Optional


# ── 기본 설정값 ──────────────────────────────────────────────────
# 모델 크기: tiny / base / small / medium / large-v3
# 정확도 우선 → small 이상, 속도 우선 → base
DEFAULT_MODEL_SIZE = "small"
DEFAULT_DEVICE     = "cpu"    # GPU 있으면 "cuda"
DEFAULT_LANGUAGE   = "ko"     # None 이면 자동 감지


@dataclass
class SpeechSegment:
    """STT 세그먼트 단위 (발화 한 토막)"""
    start: float   # 시작 시간 (초)
    end:   float   # 종료 시간 (초)
    text:  str     # 발화 텍스트


@dataclass
class WordSegment:
    """단어 단위 타임스탬프"""
    start:       float   # 단어 시작 시간 (초)
    end:         float   # 단어 종료 시간 (초)
    word:        str     # 단어
    probability: float   # 인식 신뢰도 (0~1)


def transcribe(
    video_path: str,
    model_size: str = DEFAULT_MODEL_SIZE,
    device: str = DEFAULT_DEVICE,
    language: Optional[str] = DEFAULT_LANGUAGE,
) -> List[SpeechSegment]:
    """영상 전체를 전사해 SpeechSegment 리스트로 반환한다.

    Args:
        video_path:  분석할 영상 경로
        model_size:  Whisper 모델 크기 (tiny/base/small/medium/large-v3)
        device:      실행 디바이스 (cpu / cuda)
        language:    언어 코드 (ko, en, ja ...). None 이면 자동 감지

    Returns:
        시간순으로 정렬된 SpeechSegment 리스트
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError(
            "faster-whisper가 설치되어 있지 않습니다.\n"
            "pip install faster-whisper 를 실행하세요."
        )

    print(f"[STT] 모델 로드 중: {model_size} ({device})")
    t0 = time.time()
    model = WhisperModel(model_size, device=device, download_root="./models")
    print(f"[STT] 모델 로드 완료 ({time.time() - t0:.1f}s)")

    print(f"[STT] 전사 시작: {video_path}")
    t0 = time.time()
    raw_segments, info = model.transcribe(
        video_path,
        language=language,
        beam_size=5,
        vad_filter=True,          # 무음 구간 자동 제거
        vad_parameters={
            "min_silence_duration_ms": 500,  # 0.5초 이상 무음이면 분리
        },
    )

    segments: List[SpeechSegment] = []
    for seg in raw_segments:
        text = seg.text.strip()
        if not text:
            continue
        segments.append(SpeechSegment(start=seg.start, end=seg.end, text=text))

    print(
        f"[STT] 전사 완료 ({time.time() - t0:.1f}s) — "
        f"언어: {info.language} ({info.language_probability:.0%}), "
        f"세그먼트: {len(segments)}개"
    )
    return segments


def transcribe_words(
    video_path: str,
    model_size: str = DEFAULT_MODEL_SIZE,
    device: str = DEFAULT_DEVICE,
    language: Optional[str] = DEFAULT_LANGUAGE,
) -> List[WordSegment]:
    """영상 전체를 전사해 단어 단위 타임스탬프 리스트로 반환한다."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise ImportError(
            "faster-whisper가 설치되어 있지 않습니다.\n"
            "pip install faster-whisper 를 실행하세요."
        )

    print(f"[STT] 모델 로드 중: {model_size} ({device})")
    t0 = time.time()
    model = WhisperModel(model_size, device=device, download_root="./models")
    print(f"[STT] 모델 로드 완료 ({time.time() - t0:.1f}s)")

    print(f"[STT] 단어 단위 전사 시작: {video_path}")
    t0 = time.time()
    raw_segments, info = model.transcribe(
        video_path,
        language=language,
        beam_size=5,
        word_timestamps=True,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": 500,
        },
    )

    words: List[WordSegment] = []
    for seg in raw_segments:
        if not seg.words:
            continue
        for w in seg.words:
            word = w.word.strip()
            if not word:
                continue
            words.append(WordSegment(
                start=w.start,
                end=w.end,
                word=word,
                probability=w.probability,
            ))

    print(
        f"[STT] 전사 완료 ({time.time() - t0:.1f}s) — "
        f"언어: {info.language} ({info.language_probability:.0%}), "
        f"단어: {len(words)}개"
    )
    return words


def get_scene_transcript(
    segments: List[SpeechSegment],
    start_sec: float,
    end_sec: float,
) -> str:
    """장면 시간 범위에 해당하는 발화 텍스트를 반환한다.

    장면 범위와 조금이라도 겹치는 세그먼트를 모두 포함한다.
    발화가 없는 장면이면 빈 문자열을 반환한다.

    Args:
        segments:   transcribe()가 반환한 전체 세그먼트 리스트
        start_sec:  장면 시작 시간 (초)
        end_sec:    장면 종료 시간 (초)

    Returns:
        해당 구간 발화 텍스트 (공백으로 이어붙임). 발화 없으면 ""
    """
    texts = [
        seg.text
        for seg in segments
        if seg.end > start_sec and seg.start < end_sec
    ]
    return " ".join(texts)


def has_speech(
    segments: List[SpeechSegment],
    start_sec: float,
    end_sec: float,
) -> bool:
    """해당 장면 구간에 발화가 있는지 여부만 반환한다."""
    return any(
        seg.end > start_sec and seg.start < end_sec
        for seg in segments
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("사용법: python pipeline/stt.py <영상경로> [--words]")
        print("예시:   python pipeline/stt.py inputs/video.mp4")
        print("        python pipeline/stt.py inputs/video.mp4 --words")
        sys.exit(1)

    video     = sys.argv[1]
    word_mode = "--words" in sys.argv

    if word_mode:
        # ── 단어 단위 모드 ──────────────────────────────────────
        words = transcribe_words(video)

        print(f"\n{'─' * 65}")
        print(f"{'시작':>8}  {'종료':>8}  {'신뢰도':>6}  단어")
        print(f"{'─' * 65}")
        for w in words:
            prob_bar = "█" * int(w.probability * 10)
            print(f"{w.start:>7.2f}s  {w.end:>7.2f}s  {w.probability:>5.0%}  {w.word}")
        print(f"{'─' * 65}")
        print(f"총 {len(words)}개 단어")

    else:
        # ── 세그먼트 단위 모드 (기본) ────────────────────────────
        segs = transcribe(video)

        print(f"\n{'─' * 60}")
        print(f"{'시작':>8}  {'종료':>8}  발화 내용")
        print(f"{'─' * 60}")
        for s in segs:
            print(f"{s.start:>7.1f}s  {s.end:>7.1f}s  {s.text}")
        print(f"{'─' * 60}")
        print(f"총 {len(segs)}개 세그먼트")
