"""
영상 하이라이트 자동 추출 파이프라인

사용법:
    python main.py input.mp4
    python main.py input.mp4 --top-n 3
    python main.py input.mp4 --frames-per-scene 1 --model ViT-L/14
    python main.py input.mp4 --output results.json
"""
import argparse
import json
import time

import config
from pipeline.scene_detector import detect_scenes
from pipeline.clip_scorer import score_scenes
from pipeline.highlight_selector import select_top


def parse_args():
    parser = argparse.ArgumentParser(description="영상 하이라이트 장면 자동 추출")
    parser.add_argument("video", help="분석할 mp4 영상 경로")
    parser.add_argument("--top-n", type=int, default=config.TOP_N,
                        help=f"추출할 하이라이트 장면 수 (기본값: {config.TOP_N})")
    parser.add_argument("--frames-per-scene", type=int, default=config.FRAMES_PER_SCENE,
                        help=f"장면당 추출 프레임 수 (기본값: {config.FRAMES_PER_SCENE})")
    parser.add_argument("--model", default=config.CLIP_MODEL,
                        help=f"CLIP 모델 (기본값: {config.CLIP_MODEL})")
    parser.add_argument("--frames-dir", default=config.FRAMES_DIR,
                        help=f"프레임 저장 디렉토리 (기본값: {config.FRAMES_DIR})")
    parser.add_argument("--output", default=None,
                        help="결과 JSON 저장 경로 (기본값: 표준출력)")
    return parser.parse_args()


def main():
    args = parse_args()
    total_start = time.time()

    # 1단계: 장면 분할 + 프레임 추출
    print(f"\n[1/3] 장면 분할 중: {args.video}")
    t0 = time.time()
    scenes = detect_scenes(args.video, args.frames_per_scene, args.frames_dir)
    print(f"      완료 ({time.time() - t0:.1f}s)\n")

    # 2단계: CLIP 점수 계산
    print("[2/3] CLIP 하이라이트 점수 계산 중...")
    t0 = time.time()
    scored = score_scenes(
        scenes,
        highlight_prompts=config.HIGHLIGHT_PROMPTS,
        boring_prompts=config.BORING_PROMPTS,
        model_name=args.model,
    )
    print(f"      완료 ({time.time() - t0:.1f}s)\n")

    # 3단계: 상위 N개 선택
    print(f"[3/3] 상위 {args.top_n}개 장면 선택 중...")
    t0 = time.time()
    highlights = select_top(scored, args.top_n)
    print(f"      완료 ({time.time() - t0:.1f}s)\n")

    elapsed = time.time() - total_start
    print(f"\n총 처리 시간: {elapsed:.1f}s")
    print(f"하이라이트 장면 {len(highlights)}개 선택됨\n")

    output = json.dumps(highlights, indent=2, ensure_ascii=False)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"결과 저장됨: {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
