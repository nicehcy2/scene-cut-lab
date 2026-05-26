"""
PySceneDetect로 장면 경계 감지 + 대표 프레임 추출
"""
import os
from dataclasses import dataclass
from typing import List

from scenedetect import detect, ContentDetector, open_video, save_images


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

    scene_list = detect(video_path, ContentDetector())
    if not scene_list:
        raise RuntimeError("감지된 장면이 없습니다. 영상 파일을 확인하세요.")

    print(f"[SceneDetect] {len(scene_list)}개 장면 감지됨")

    # {Scene: [frame_path, ...]} 형태로 반환
    scene_to_paths = save_images(
        scene_list,
        open_video(video_path),
        num_images=frames_per_scene,
        output_dir=frames_dir,
    )

    scenes = []
    for i, scene in enumerate(scene_list):
        start_tc, end_tc = scene
        raw_paths = scene_to_paths.get(i, [])
        frame_paths = [os.path.join(frames_dir, p) for p in raw_paths]
        scenes.append(Scene(
            index=i + 1,
            start=start_tc.get_timecode(),
            end=end_tc.get_timecode(),
            start_sec=start_tc.get_seconds(),
            end_sec=end_tc.get_seconds(),
            frame_paths=frame_paths,
        ))

    return scenes
