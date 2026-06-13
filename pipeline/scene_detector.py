"""
PySceneDetect로 장면 경계 감지 + 대표 프레임 추출

ContentDetector 기본값 대신 AdaptiveDetector를 SceneManager에 연결해 사용한다.
적응형 임계값은 카메라 이동/손떨림/조명 변화에 강해 브이로그 과분할을 줄인다.
튜닝 값은 config.SCENE_* 참고.
"""
import os
import time
from dataclasses import dataclass
from typing import List

from scenedetect import (
    open_video,
    save_images,
    SceneManager,
    StatsManager,
    ContentDetector,
)

import config


@dataclass
class Scene:
    index: int
    start: str       # "HH:MM:SS.mmm"
    end: str
    start_sec: float
    end_sec: float
    frame_paths: List[str]


def detect_scenes(video_path: str, frames_per_scene: int, frames_dir: str) -> List[Scene]:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {video_path}")

    os.makedirs(frames_dir, exist_ok=True)

    t0 = time.time()
    video = open_video(video_path)

    stats_manager = StatsManager()
    scene_manager = SceneManager(stats_manager)
    scene_manager.add_detector(ContentDetector(
        threshold=config.SCENE_THRESHOLD,
        min_scene_len=f"{config.SCENE_MIN_LEN_SEC}s",
    ))
    print(
        f"[SceneDetect] ContentDetector "
        f"(threshold={config.SCENE_THRESHOLD}, "
        f"min_scene_len={config.SCENE_MIN_LEN_SEC}s)"
    )

    scene_manager.detect_scenes(video, show_progress=False)
    scene_list = scene_manager.get_scene_list()
    if not scene_list:
        raise RuntimeError("감지된 장면이 없습니다. 영상 파일을 확인하세요.")
    print(f"[SceneDetect] 장면 감지 완료 ({time.time() - t0:.1f}s) — {len(scene_list)}개 장면")

    # 컷 지표 저장 (임계값 튜닝용). 실패해도 파이프라인은 계속 진행.
    stats_path = os.path.join(os.path.dirname(frames_dir) or ".", "scene_stats.csv")
    try:
        stats_manager.save_to_csv(stats_path)
        print(f"[SceneDetect] 컷 지표 저장: {stats_path}")
    except Exception as e:
        print(f"[SceneDetect] 컷 지표 저장 건너뜀: {e}")

    # save_images는 처음부터 다시 읽으므로 새 비디오 스트림을 넘긴다.
    t0 = time.time()
    scene_to_paths = save_images(
        scene_list,
        open_video(video_path),
        num_images=frames_per_scene,
        output_dir=frames_dir,
    )
    print(f"[SceneDetect] 프레임 추출 완료 ({time.time() - t0:.1f}s)")

    scenes = []
    for i, scene in enumerate(scene_list):
        start_tc, end_tc = scene
        raw_paths = scene_to_paths.get(i, [])
        frame_paths = [os.path.join(frames_dir, p) for p in raw_paths]
        duration = end_tc.get_seconds() - start_tc.get_seconds()
        print(
            f"  Scene {i + 1:3d} | {start_tc.get_timecode()} ~ {end_tc.get_timecode()} "
            f"| {duration:.1f}s | 프레임 {len(frame_paths)}장"
        )
        scenes.append(Scene(
            index=i + 1,
            start=start_tc.get_timecode(),
            end=end_tc.get_timecode(),
            start_sec=start_tc.get_seconds(),
            end_sec=end_tc.get_seconds(),
            frame_paths=frame_paths,
        ))

    return scenes
