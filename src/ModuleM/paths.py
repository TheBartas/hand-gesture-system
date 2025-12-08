from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / 'data'
INFO_PATH = DATA_PATH / 'hand_gesture_info.json'
STATIC_PATH = DATA_PATH / 'static'
DYNAMIC_PATH = DATA_PATH / 'dynamic'
MODEL_KERAS_PATH = PROJECT_ROOT / 'model' / 'model.keras'
MODEL_TFLITE_PATH = PROJECT_ROOT / 'model' / 'model.tflite'
MODEL_JSON_PATH = PROJECT_ROOT / 'model' / 'model.json'