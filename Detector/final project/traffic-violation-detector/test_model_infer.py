import sys
from pathlib import Path
import numpy as np

print('PYTHON:', sys.executable)

try:
    from ultralytics import YOLO
except Exception as e:
    print('ULTRALYTICS_IMPORT_ERROR:', e)
    raise

models_dir = Path(__file__).resolve().parents[1] / 'models'
pt_files = sorted(models_dir.glob('*.pt'))
if not pt_files:
    print('No .pt models found in', models_dir)
    sys.exit(2)

model_path = pt_files[0]
print('Using model:', model_path)

model = YOLO(str(model_path))
print('Model loaded; names count:', len(getattr(model, 'names', [])))

# create a dummy image (640x480) and run one inference
img = np.zeros((480, 640, 3), dtype=np.uint8)
res = model(img)
print('Inference result type:', type(res), 'len:', len(res))
if len(res) > 0:
    r = res[0]
    boxes = getattr(r, 'boxes', None)
    if boxes is not None:
        try:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            classes = boxes.cls.cpu().numpy().astype(int)
            print('Boxes:', xyxy.shape)
            print('Confs sample:', confs[:5])
        except Exception:
            print('Boxes object present but conversion failed, printing repr:')
            print(repr(boxes))
    else:
        print('No boxes in result')

print('TEST DONE')
