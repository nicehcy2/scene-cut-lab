CLIP_MODEL = "ViT-B/32"

HIGHLIGHT_PROMPTS = [
    "a man with glasses speaking directly to the camera",
    "a man wearing glasses talking and presenting",
    "close-up of a man with glasses explaining something",
    "a man with glasses looking at the camera and speaking",
]

BORING_PROMPTS = [
    "a woman appearing in the scene",
    "a female person on screen",
    "woman talking or walking in the video",
    "scene without a man with glasses",
    "dark or blurry footage",
]

FRAMES_PER_SCENE = 3
TOP_N = 5
FRAMES_DIR = "frames"
