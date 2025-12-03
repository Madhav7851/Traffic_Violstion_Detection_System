import sys
import traceback

print('PYTHON:', sys.executable)
try:
    from ultralytics import YOLO
    print('YOLO_IMPORT_OK')
except Exception as e:
    print('YOLO_IMPORT_ERROR:', repr(e))
    traceback.print_exc()
