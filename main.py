"""
영상 하이라이트 자동 추출 파이프라인

[전체 자동 모드] — 기본. 장면 감지 + STT 전사 + Gemini 점수화 + 하이라이트 선택 + (선택) 렌더링
    python main.py input.mp4

[장면 감지만 실행] — Gemini 호출 없이 장면 분할과 그리드 이미지만 생성. 분할 결과 확인 용도
    python main.py input.mp4 --detect-only

[선택/렌더링만 실행] — 전체 자동 모드로 생성된 results.json을 받아 top-N 조정이나 재렌더링
    python main.py input.mp4 --from-scores runs/.../results.json
"""
import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import config
from pipeline.scene_detector import detect_scenes
from pipeline.gemini_scorer_v2 import score_scenes_v2
from pipeline.stt import transcribe
from pipeline.highlight_selector import select_top
from pipeline.video_exporter import export_highlight
from pipeline.inspect.meta_extractor import compute_meta
from pipeline.inspect.grid_builder import build_grids


def _make_run_dir(video_path: str) -> str:
    stem = Path(video_path).stem
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", f"{stem}_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def parse_args():
    parser = argparse.ArgumentParser(description="영상 하이라이트 장면 자동 추출")
    parser.add_argument("video", help="분석할 mp4 영상 경로 (inputs/ 폴더 기준)")

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--detect-only", action="store_true",
                      help="장면 감지 + 프레임 추출만 실행하고 scenes.json 저장.")
    mode.add_argument("--from-scores", metavar="SCORED_JSON",
                      help="전체 자동 모드로 생성된 results.json을 받아 선택 + 영상 합치기만 실행 (top-n 조정이나 재렌더링 용도)")

    parser.add_argument("--subject", default=None, choices=config.SUBJECT_CHOICES,
                        metavar="SUBJECT",
                        help=f"피사체 유형 (전체 자동 모드 전용). "
                             f"선택지: {', '.join(config.SUBJECT_CHOICES)}")
    parser.add_argument("--top-n", type=int, default=config.TOP_N,
                        help="추출할 하이라이트 장면 수 (기본값: 0 = 통과한 장면 전체 포함)")
    parser.add_argument("--frames-per-scene", type=int, default=config.FRAMES_PER_SCENE,
                        help=f"장면당 추출 프레임 수 (기본값: {config.FRAMES_PER_SCENE})")
    parser.add_argument("--model", default=config.GEMINI_MODEL,
                        help=f"Gemini 모델 (기본값: {config.GEMINI_MODEL})")
    parser.add_argument("--no-export", action="store_true",
                        help="하이라이트 영상 생성 건너뜀 (기본: 영상 생성)")
    parser.add_argument("--output", default=None, metavar="JSON_PATH",
                        help="결과 JSON 저장 경로 (기본값: runs/{영상명}_{시각}/results.json)")
    parser.add_argument("--include-maybe", action="store_true",
                        help="maybe 장면도 포함 (기본: keep 장면만)")
    parser.add_argument("--maybe-min-score", type=float, default=config.MAYBE_MIN_SCORE,
                        help=f"maybe 포함 최소 점수 (기본값: {config.MAYBE_MIN_SCORE}, 범위: 0~100)")
    parser.add_argument("--no-stt", action="store_true",
                        help="STT 전사 건너뜀 (발화 정보 없이 시각 정보만으로 평가)")
    parser.add_argument("--stt-model", default="small",
                        choices=["tiny", "base", "small", "medium", "large-v3"],
                        help="Whisper 모델 크기 (기본값: small / 한국어 브이로그 권장: medium)")
    parser.add_argument("--style", default="장면 중심",
                        choices=["장면 중심", "음성 중심", "균형"],
                        help="점수 가중치 스타일 (기본값: 장면 중심)")
    return parser.parse_args()


# ────────────────────────────────────────────────
# 모드 1: 장면 감지만 실행
# ────────────────────────────────────────────────
def run_detect_only(args, run_dir):
    frames_dir = os.path.join(run_dir, "frames")
    grids_dir = os.path.join(run_dir, "grids")
    print(f"\n실행 디렉토리: {run_dir}")

    print(f"\n[1/3] 장면 분할 중: {args.video}")
    t0 = time.time()
    scenes = detect_scenes(args.video, args.frames_per_scene, frames_dir)
    print(f"      완료 ({time.time() - t0:.1f}s) — {len(scenes)}개 장면\n")

    print(f"[2/3] 메타데이터 추출 중...")
    t0 = time.time()
    metas = compute_meta(scenes)
    print(f"      완료 ({time.time() - t0:.1f}s)\n")

    print(f"[3/3] 이미지 그리드 생성 중...")
    t0 = time.time()
    grid_paths = build_grids(scenes, grids_dir)
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

    json_path = args.output or os.path.join(run_dir, "scenes.json")
    _save(output_data, json_path)
    print(f"\ngrids/ 폴더의 그리드 이미지로 장면 분할 결과를 확인하세요.")
    print(f"문제가 없으면 아래 명령으로 전체 분석을 이어서 실행하세요:")
    print(f"  python main.py \"{args.video}\" --export")


# ────────────────────────────────────────────────
# 모드 2: scored.json 받아서 선택 + 합치기
# ────────────────────────────────────────────────
def run_from_scores(args, run_dir):
    total_start = time.time()
    print(f"\n실행 디렉토리: {run_dir}")

    print(f"\n[1/3] scored.json 로드 중: {args.from_scores}")
    t0 = time.time()
    with open(args.from_scores, encoding="utf-8") as f:
        scored = json.load(f)
    print(f"      완료 ({time.time() - t0:.1f}s) — {len(scored)}개 장면 로드됨\n")

    top_label = f"상위 {args.top_n}개" if args.top_n > 0 else "전체"
    print(f"[2/3] 장면 선택 중 ({top_label})...")
    t0 = time.time()
    highlights = select_top(scored, args.top_n, keep_only=not args.include_maybe, maybe_min_score=args.maybe_min_score)
    print(f"      완료 ({time.time() - t0:.1f}s) — {len(highlights)}개 선택됨")
    for h in highlights:
        print(f"  Scene {h['scene']:3d} | {h['start']} ~ {h['end']} | score={h.get('final_score', '?')}")
    print()

    if not args.no_export:
        export_path = os.path.join(run_dir, "highlight.mp4")
        print(f"[3/3] 영상 합치는 중: {export_path}")
        t0 = time.time()
        export_highlight(args.video, highlights, export_path)
        size_mb = os.path.getsize(export_path) / (1024 * 1024)
        print(f"      완료 ({time.time() - t0:.1f}s) — {size_mb:.1f} MB\n")
    else:
        print("[3/3] --no-export 지정, 영상 파일 생성 건너뜀\n")

    print(f"총 처리 시간: {time.time() - total_start:.1f}s")
    _save(highlights, args.output or os.path.join(run_dir, "results.json"))


# ────────────────────────────────────────────────
# 모드 3: 전체 자동 (Gemini + STT)
# ────────────────────────────────────────────────
def run_full_auto(args, run_dir):
    total_start = time.time()
    frames_dir = os.path.join(run_dir, "frames")
    subject_label = f" [{args.subject}]" if args.subject else " [자동 감지]"

    print(f"\n실행 디렉토리: {run_dir}")
    print(f"\n[1/4] 장면 분할 중: {args.video}")
    t0 = time.time()
    scenes = detect_scenes(args.video, args.frames_per_scene, frames_dir)
    print(f"      완료 ({time.time() - t0:.1f}s) — {len(scenes)}개 장면\n")

    stt_segments = []
    if args.no_stt:
        print(f"[2/4] STT 전사 건너뜀 (--no-stt)\n")
    else:
        print(f"[2/4] STT 전사 중 (모델: {args.stt_model})...")
        t0 = time.time()
        stt_segments = transcribe(args.video, model_size=args.stt_model)
        print(f"      완료 ({time.time() - t0:.1f}s)\n")

    print(f"[3/4] Gemini 하이라이트 점수 계산 중{subject_label} [스타일: {args.style}]...")
    t0 = time.time()
    scored = score_scenes_v2(
        scenes,
        stt_segments=stt_segments,
        model_name=args.model,
        subject=args.subject,
        style=args.style,
    )
    print(f"      완료 ({time.time() - t0:.1f}s)\n")

    print(f"[4/4] 장면 선택 중...")
    t0 = time.time()
    highlights = select_top(scored, args.top_n, keep_only=not args.include_maybe, maybe_min_score=args.maybe_min_score)
    print(f"      완료 ({time.time() - t0:.1f}s) — {len(highlights)}개 선택됨\n")

    if not args.no_export:
        export_path = os.path.join(run_dir, "highlight.mp4")
        print(f"[영상 합치기] {export_path}")
        t0 = time.time()
        export_highlight(args.video, highlights, export_path)
        size_mb = os.path.getsize(export_path) / (1024 * 1024)
        print(f"      완료 ({time.time() - t0:.1f}s) — {size_mb:.1f} MB\n")

    print(f"총 처리 시간: {time.time() - total_start:.1f}s")
    _save(highlights, args.output or os.path.join(run_dir, "results.json"))


# ────────────────────────────────────────────────
# 공통 유틸
# ────────────────────────────────────────────────
def _save(data, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"저장됨: {path}")


def main():
    args = parse_args()
    if not os.path.dirname(args.video):
        args.video = os.path.join("inputs", args.video)
    run_dir = _make_run_dir(args.video)
    if args.detect_only:
        run_detect_only(args, run_dir)
    elif args.from_scores:
        run_from_scores(args, run_dir)
    else:
        run_full_auto(args, run_dir)


if __name__ == "__main__":
    main()
