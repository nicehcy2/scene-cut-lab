"""
영상 하이라이트 자동 추출 파이프라인

[전체 자동 모드]
    python main.py input.mp4 --subject 사람 --export highlight.mp4

[2단계 분리 모드]
  1단계 - 장면 감지만 실행, GPT용 JSON 저장:
    python main.py input.mp4 --detect-only --output scenes.json

  2단계 - GPT가 채운 scored.json으로 영상 합치기:
    python main.py input.mp4 --from-scores scored.json --export highlight.mp4
"""
import argparse
import json
import os
import time

import config
from pipeline.scene_detector import detect_scenes, Scene
from pipeline.gemini_scorer import score_scenes
from pipeline.highlight_selector import select_top
from pipeline.video_exporter import export_highlight
from pipeline.meta_extractor import compute_meta
from pipeline.grid_builder import build_grids


def parse_args():
    parser = argparse.ArgumentParser(description="영상 하이라이트 장면 자동 추출")
    parser.add_argument("video", help="분석할 mp4 영상 경로")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--detect-only", action="store_true",
                      help="장면 감지 + 프레임 추출만 실행하고 scenes.json 저장. "
                           "GPT에게 넘겨 점수를 받은 뒤 --from-scores로 이어서 실행")
    mode.add_argument("--from-scores", metavar="SCORED_JSON",
                      help="GPT가 점수 매긴 JSON 파일을 받아 선택 + 영상 합치기만 실행")

    parser.add_argument("--subject", default=None, choices=config.SUBJECT_CHOICES,
                        metavar="SUBJECT",
                        help=f"피사체 유형 (전체 자동 모드 전용). "
                             f"선택지: {', '.join(config.SUBJECT_CHOICES)}")
    parser.add_argument("--top-n", type=int, default=config.TOP_N,
                        help=f"추출할 하이라이트 장면 수 (기본값: {config.TOP_N})")
    parser.add_argument("--frames-per-scene", type=int, default=config.FRAMES_PER_SCENE,
                        help=f"장면당 추출 프레임 수 (기본값: {config.FRAMES_PER_SCENE})")
    parser.add_argument("--model", default=config.GEMINI_MODEL,
                        help=f"Gemini 모델 (기본값: {config.GEMINI_MODEL})")
    parser.add_argument("--frames-dir", default=config.FRAMES_DIR,
                        help=f"프레임 저장 디렉토리 (기본값: {config.FRAMES_DIR})")
    parser.add_argument("--export", default=None, metavar="OUTPUT_VIDEO",
                        help="하이라이트 영상 출력 경로 (예: highlight.mp4)")
    parser.add_argument("--output", default=None, metavar="JSON_PATH",
                        help="결과 JSON 저장 경로 (기본값: 표준출력)")
    parser.add_argument("--keep-only", action="store_true",
                        help="keep 장면만 포함 (maybe 제외)")
    return parser.parse_args()


# ────────────────────────────────────────────────
# 모드 1: 장면 감지만 실행
# ────────────────────────────────────────────────
def run_detect_only(args):
    print(f"\n[1/3] 장면 분할 중: {args.video}")
    t0 = time.time()
    scenes = detect_scenes(args.video, args.frames_per_scene, args.frames_dir)
    print(f"      완료 ({time.time() - t0:.1f}s) — {len(scenes)}개 장면\n")

    print(f"[2/3] 메타데이터 추출 중...")
    t0 = time.time()
    metas = compute_meta(scenes)
    print(f"      완료 ({time.time() - t0:.1f}s)\n")

    print(f"[3/3] 이미지 그리드 생성 중...")
    t0 = time.time()
    grid_paths = build_grids(scenes)
    print(f"      완료 ({time.time() - t0:.1f}s) — {len(grid_paths)}장\n")

    output_data = [
        {
            "scene": s.index,
            "start": s.start,
            "end": s.end,
            "start_sec": s.start_sec,
            "end_sec": s.end_sec,
            "duration_sec": round(s.end_sec - s.start_sec, 3),
            "frame_paths": s.frame_paths,
            "meta": metas[i],
        }
        for i, s in enumerate(scenes)
    ]

    json_path = args.output or "scenes.json"
    _save_or_print(output_data, json_path)
    print(
        "\nGPT에게 넘길 준비 완료.\n"
        f"  → {json_path}\n"
        f"  → grids/ 폴더 ({len(grid_paths)}장)\n\n"
        "점수를 받아 scored.json으로 저장한 뒤 아래 명령으로 이어서 실행하세요:\n\n"
        f"  python main.py \"{args.video}\" --from-scores scored.json --top-n 0 --export highlight.mp4"
    )


# ────────────────────────────────────────────────
# 모드 2: scored.json 받아서 선택 + 합치기
# ────────────────────────────────────────────────
def run_from_scores(args):
    total_start = time.time()

    print(f"\n[1/3] scored.json 로드 중: {args.from_scores}")
    t0 = time.time()
    with open(args.from_scores, encoding="utf-8") as f:
        scored = json.load(f)
    print(f"      완료 ({time.time() - t0:.1f}s) — {len(scored)}개 장면 로드됨\n")

    top_label = f"상위 {args.top_n}개" if args.top_n > 0 else "전체"
    print(f"[2/3] 장면 선택 중 ({top_label})...")
    t0 = time.time()
    highlights = select_top(scored, args.top_n, keep_only=args.keep_only)
    print(f"      완료 ({time.time() - t0:.1f}s) — {len(highlights)}개 선택됨")
    for h in highlights:
        print(f"  Scene {h['scene']:3d} | {h['start']} ~ {h['end']} | score={h.get('final_score', '?')}")
    print()

    if args.export:
        print(f"[3/3] 영상 합치는 중: {args.export}")
        t0 = time.time()
        export_highlight(args.video, highlights, args.export)
        size_mb = os.path.getsize(args.export) / (1024 * 1024)
        print(f"      완료 ({time.time() - t0:.1f}s) — {size_mb:.1f} MB\n")
    else:
        print("[3/3] --export 미지정, 영상 파일 생성 건너뜀\n")

    print(f"총 처리 시간: {time.time() - total_start:.1f}s")
    _save_or_print(highlights, args.output)


# ────────────────────────────────────────────────
# 모드 3: 전체 자동 (Gemini)
# ────────────────────────────────────────────────
def run_full_auto(args):
    total_start = time.time()
    subject_label = f" [{args.subject}]" if args.subject else " [자동 감지]"

    print(f"\n[1/4] 장면 분할 중: {args.video}")
    t0 = time.time()
    scenes = detect_scenes(args.video, args.frames_per_scene, args.frames_dir)
    print(f"      완료 ({time.time() - t0:.1f}s)\n")

    print(f"[2/4] Gemini 하이라이트 점수 계산 중{subject_label}...")
    t0 = time.time()
    scored = score_scenes(
        scenes,
        master_prompt=config.MASTER_PROMPT,
        subject_prompts=config.SUBJECT_PROMPTS,
        model_name=args.model,
        subject=args.subject,
    )
    print(f"      완료 ({time.time() - t0:.1f}s)\n")

    print(f"[3/4] 상위 {args.top_n}개 장면 선택 중...")
    t0 = time.time()
    highlights = select_top(scored, args.top_n, keep_only=args.keep_only)
    print(f"      완료 ({time.time() - t0:.1f}s)\n")

    if args.export:
        print(f"[4/4] 하이라이트 영상 생성 중: {args.export}")
        t0 = time.time()
        export_highlight(args.video, highlights, args.export)
        size_mb = os.path.getsize(args.export) / (1024 * 1024)
        print(f"      완료 ({time.time() - t0:.1f}s) — {size_mb:.1f} MB\n")
    else:
        print("[4/4] --export 미지정, 영상 파일 생성 건너뜀\n")

    print(f"총 처리 시간: {time.time() - total_start:.1f}s")
    print(f"하이라이트 장면 {len(highlights)}개 선택됨\n")
    _save_or_print(highlights, args.output)


# ────────────────────────────────────────────────
# 공통 유틸
# ────────────────────────────────────────────────
def _save_or_print(data, path=None):
    text = json.dumps(data, indent=2, ensure_ascii=False)
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"저장됨: {path}")
    else:
        print(text)


def main():
    args = parse_args()
    if args.detect_only:
        run_detect_only(args)
    elif args.from_scores:
        run_from_scores(args)
    else:
        run_full_auto(args)


if __name__ == "__main__":
    main()
