"""
하이라이트 장면을 잘라 하나의 영상으로 합치는 모듈.
FFmpeg 바이너리는 imageio-ffmpeg에서 가져온다.
"""
import os
import subprocess
import tempfile
from typing import List

import imageio_ffmpeg


def export_highlight(
    video_path: str,
    highlights: List[dict],
    output_path: str,
) -> str:
    """하이라이트 장면을 잘라 하나의 mp4로 합친 뒤 output_path에 저장한다."""
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    with tempfile.TemporaryDirectory() as tmp_dir:
        segment_paths = _cut_segments(ffmpeg, video_path, highlights, tmp_dir)
        _concat_segments(ffmpeg, segment_paths, output_path)

    return output_path


def _cut_segments(
    ffmpeg: str,
    video_path: str,
    highlights: List[dict],
    tmp_dir: str,
) -> List[str]:
    """각 하이라이트 구간을 개별 파일로 잘라낸다."""
    paths = []
    for i, h in enumerate(highlights):
        out = os.path.join(tmp_dir, f"seg_{i:03d}.mp4")
        cmd = [
            ffmpeg, "-y",
            "-ss", h["start"],
            "-to", h["end"],
            "-i", video_path,
            "-c:v", "libx264",    # 프레임 단위 정확한 컷
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",  # Windows 호환성
            "-c:a", "aac",
            out,
        ]
        _run(cmd)
        paths.append(out)
    return paths


def _concat_segments(ffmpeg: str, segment_paths: List[str], output_path: str) -> None:
    """세그먼트 목록을 하나의 파일로 이어붙인다."""
    # FFmpeg concat demuxer 방식: 중간 재인코딩 없이 빠르게 합침
    list_path = os.path.join(os.path.dirname(segment_paths[0]), "concat_list.txt")
    with open(list_path, "w", encoding="utf-8") as f:
        for path in segment_paths:
            # Windows 경로의 역슬래시를 슬래시로 변환
            f.write(f"file '{path.replace(os.sep, '/')}'\n")

    cmd = [
        ffmpeg, "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_path,
        "-c", "copy",
        output_path,
    ]
    _run(cmd)


def _run(cmd: List[str]) -> None:
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"FFmpeg 오류:\n{result.stderr.decode(errors='replace')}"
        )
