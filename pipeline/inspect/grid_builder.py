"""
장면 대표 프레임 이미지 그리드 생성 (GPT 수동 채점용)

출력 예시:
  grids/grid_001_012.jpg  → Scene 1~12
  grids/grid_013_024.jpg  → Scene 13~24
  grids/grid_025_037.jpg  → Scene 25~37

각 행 레이아웃:
  ┌─────────────────────────────────────────────────────┐
  │ Scene  1  |  00:00:00.000 ~ 00:00:36.500  |  36.5s  │  ← 라벨
  │ [frame 1] │ [frame 2]  │ [frame 3]                  │  ← 썸네일
  └─────────────────────────────────────────────────────┘
"""
import os
from typing import List

from PIL import Image, ImageDraw, ImageFont

from pipeline.scene_detector import Scene

# ── 레이아웃 상수 ────────────────────────────────────────
THUMB_W = 320        # 썸네일 너비 (px)
THUMB_H = 180        # 썸네일 높이 (px, 16:9)
LABEL_H = 38         # 장면 라벨 높이
CELL_PAD = 4         # 셀 간 여백
SCENES_PER_GRID = 12 # 그리드당 장면 수

BG_COLOR     = (24,  24,  24 )  # 전체 배경
LABEL_BG     = (48,  48,  48 )  # 라벨 배경
TEXT_COLOR   = (220, 220, 220)  # 라벨 텍스트
BORDER_COLOR = (72,  72,  72 )  # 썸네일 테두리
EMPTY_COLOR  = (40,  40,  40 )  # 프레임 없을 때 배경


def build_grids(
    scenes: List[Scene],
    output_dir: str = "grids",
    scenes_per_grid: int = SCENES_PER_GRID,
) -> List[str]:
    """장면 목록을 그리드 이미지로 저장하고 파일 경로 목록 반환."""
    os.makedirs(output_dir, exist_ok=True)
    font = _load_font(13)

    chunks = [scenes[i:i + scenes_per_grid] for i in range(0, len(scenes), scenes_per_grid)]
    grid_paths = []

    for chunk in chunks:
        first, last = chunk[0].index, chunk[-1].index
        out_path = os.path.join(output_dir, f"grid_{first:03d}_{last:03d}.jpg")

        img = _render_grid(chunk, font)
        img.save(out_path, quality=88)
        grid_paths.append(out_path)
        print(f"  [Grid] {out_path}  (Scene {first}~{last}, {img.width}x{img.height}px)")

    return grid_paths


def _render_grid(scenes: List[Scene], font) -> Image.Image:
    # 장면 중 최대 프레임 수 기준으로 열 수 결정
    n_cols = max((len(s.frame_paths) for s in scenes), default=3)
    n_cols = max(n_cols, 1)

    row_w = CELL_PAD + n_cols * (THUMB_W + CELL_PAD)
    row_h = LABEL_H + THUMB_H + CELL_PAD
    canvas = Image.new("RGB", (row_w, CELL_PAD + len(scenes) * row_h), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    for row_idx, scene in enumerate(scenes):
        y = CELL_PAD + row_idx * row_h

        # 라벨
        draw.rectangle([CELL_PAD, y, row_w - CELL_PAD, y + LABEL_H - 2], fill=LABEL_BG)
        duration = scene.end_sec - scene.start_sec
        label = f"Scene {scene.index:3d}  |  {scene.start} ~ {scene.end}  |  {duration:.1f}s"
        draw.text((CELL_PAD + 8, y + 10), label, font=font, fill=TEXT_COLOR)

        # 썸네일
        for col_idx in range(n_cols):
            x = CELL_PAD + col_idx * (THUMB_W + CELL_PAD)
            fy = y + LABEL_H

            if col_idx < len(scene.frame_paths):
                try:
                    thumb = Image.open(scene.frame_paths[col_idx]).convert("RGB")
                    thumb = thumb.resize((THUMB_W, THUMB_H), Image.LANCZOS)
                    canvas.paste(thumb, (x, fy))
                except Exception:
                    _draw_empty(draw, x, fy, font)
            else:
                _draw_empty(draw, x, fy, font)

            draw.rectangle(
                [x, fy, x + THUMB_W - 1, fy + THUMB_H - 1],
                outline=BORDER_COLOR,
            )

    return canvas


def _draw_empty(draw: ImageDraw.ImageDraw, x: int, y: int, font) -> None:
    draw.rectangle([x, y, x + THUMB_W - 1, y + THUMB_H - 1], fill=EMPTY_COLOR)
    draw.text((x + THUMB_W // 2 - 25, y + THUMB_H // 2 - 7), "no frame",
              font=font, fill=(90, 90, 90))


def _load_font(size: int):
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
