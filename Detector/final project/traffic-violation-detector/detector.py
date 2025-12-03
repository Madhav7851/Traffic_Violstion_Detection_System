import time
import json
import logging
import os
from pathlib import Path
from datetime import datetime

import cv2
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

from config import get_settings
from services.firebase_service import FirebaseService
from services.redis_service import RedisService
from db import init_db, get_session, create_violation, create_challan
from anpr import ocr_plate_image, crop_plate_from_frame

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# --- Basic violation configuration and helpers ---
VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorbike', 'motorcycle'}  # common names in COCO-like models

# Simple fine mapping (Python dict) - adjust amounts as needed
FINES = {
    'no_helmet': 500.0,
    'vehicle_detected': 1000.0,
    'no_plate': 1500.0,
    'wrong_lane': 700.0,
}


def create_duplicate_key(cls_name: str, bbox: tuple) -> str:
    x1, y1, x2, y2 = map(int, bbox)
    return f"{cls_name}:{x1}-{y1}-{x2}-{y2}"


def _bbox_center(bbox):
    x1, y1, x2, y2 = map(int, bbox)
    return (x1 + x2) // 2, (y1 + y2) // 2


def _iou(a, b):
    ax1, ay1, ax2, ay2 = map(int, a)
    bx1, by1, bx2, by2 = map(int, b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def detect_no_helmet(detections):
    """Return list of (motorbike_bbox, person_bbox) that appear to have no helmet."""
    bikes = []
    people = []
    helmets = []
    for name, conf, bbox in detections:
        lname = name.lower()
        if 'helmet' in lname:
            helmets.append(bbox)
        elif 'person' in lname:
            people.append(bbox)
        elif lname in ('motorbike', 'motorcycle'):
            bikes.append(bbox)

    results = []
    for bike in bikes:
        for person in people:
            if _iou(bike, person) > 0.05:
                has_helmet = any(_iou(person, h) > 0.1 for h in helmets)
                if not has_helmet:
                    results.append((bike, person))
    return results


def detect_wrong_lane(bbox, frame_shape, forbidden_region=(0.0, 0.3)):
    h, w = frame_shape[:2]
    cx, cy = _bbox_center(bbox)
    nx = cx / float(w) if w > 0 else 0.0
    xmin, xmax = forbidden_region
    return xmin <= nx <= xmax


def call_alert_stub(violation: dict, outpath: Path):
    entry = violation.copy()
    entry['alerted_at'] = datetime.utcnow().isoformat()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open('a', encoding='utf8') as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info("Call/alert logged to %s", outpath)


def main():
    settings = get_settings()
    firebase = FirebaseService(credentials_path=settings.FIREBASE_CREDS_PATH,
                               database_url=settings.FIREBASE_DATABASE_URL)
    redis = RedisService(host=settings.REDIS_HOST, port=settings.REDIS_PORT, password=settings.REDIS_PASSWORD)

    # allow disabling model load for smoke/runs where torch/ultralytics are unavailable
    disable_model = str(os.getenv('DISABLE_MODEL', '')).lower() in ('1', 'true', 'yes')
    models_dir = Path(__file__).parents[1] / 'models'
    models_list = []
    if not disable_model and YOLO is not None:
        # autodiscover .pt models in the models directory
        if models_dir.exists():
            for p in sorted(models_dir.glob('*.pt')):
                try:
                    logger.info("Loading model from %s", p)
                    m = YOLO(str(p))
                    models_list.append((p.name, m))
                except Exception as e:
                    logger.error("Failed to load model %s: %s", p, e)
        if not models_list:
            logger.warning("No models loaded from %s; detector will run without ML.", models_dir)
    else:
        if disable_model:
            logger.info("Model loading disabled via DISABLE_MODEL env var. Running in scaffold mode.")
        elif YOLO is None:
            logger.warning("ultralytics YOLO not available. Detector will use frame capture only (no ML).")
        else:
            logger.warning("Models directory not found at %s. Detector will use frame capture only (no ML).", models_dir)

    # primary convenience variable for older code paths
    model = models_list[0][1] if models_list else None

    video_src = settings.VIDEO_SOURCE
    try:
        cap = cv2.VideoCapture(int(video_src)) if str(video_src).isdigit() else cv2.VideoCapture(video_src)
    except Exception:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        logger.error("Failed to open video source: %s", video_src)
        return

    logger.info("Started video capture from %s", video_src)
    call_log = Path(__file__).parents[1] / 'captured_frames' / 'call_log.jsonl'
    frame_counter = 0

    # simple in-memory buffers/state
    seen_counters = {}
    ocr_buffers = {}

    # smoke-run controls via environment variables
    max_frames_env = os.getenv('MAX_FRAMES')
    try:
        max_frames = int(max_frames_env) if max_frames_env is not None else 0
    except Exception:
        max_frames = 0

    nogui = str(os.getenv('NO_GUI', '')).lower() in ('1', 'true', 'yes')

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("No frame read (end of stream?)")
                break

            frame_counter += 1

            detections = []
            # run each loaded model and merge detections
            if models_list:
                for mname, mobj in models_list:
                    try:
                        results = mobj(frame, conf=settings.CONFIDENCE_THRESHOLD, verbose=False)
                        det = results[0]
                        if det.boxes is not None:
                            for box in det.boxes:
                                coords = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else box.xyxy[0]
                                x1, y1, x2, y2 = coords.tolist()
                                cls_id = int(box.cls[0])
                                conf = float(box.conf[0])
                                name = mobj.names.get(cls_id, str(cls_id)) if hasattr(mobj, 'names') else f"{mname}:{cls_id}"
                                detections.append((name, conf, (x1, y1, x2, y2)))
                    except Exception as e:
                        logger.debug("Model %s inference failed: %s", mname, e)

            # derive helmet violations quickly
            helmet_issues = detect_no_helmet(detections)
            for bike_bbox, person_bbox in helmet_issues:
                key = create_duplicate_key('no_helmet', bike_bbox)
                if redis.is_duplicate(key):
                    continue
                redis.mark_seen(key, ttl=10)
                x1, y1, x2, y2 = map(int, bike_bbox)
                violation = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'camera_id': 'CAM_GENERIC_01',
                    'violation_type': 'no_helmet',
                    'class': 'motorbike',
                    'confidence': 1.0,
                    'bbox': [x1, y1, x2, y2]
                }
                snap_dir = Path(__file__).parents[1] / 'captured_frames'
                snap_dir.mkdir(parents=True, exist_ok=True)
                snap_path = snap_dir / f"violation_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}.jpg"
                roi = frame[y1:y2, x1:x2]
                if roi is None or roi.size == 0:
                    cv2.imwrite(str(snap_path), frame)
                else:
                    cv2.imwrite(str(snap_path), roi)
                violation['snapshot'] = str(snap_path)
                try:
                    vid = firebase.save_violation(violation)
                    violation['violation_id'] = vid
                except Exception as e:
                    logger.error("Failed to save helmet violation to firebase: %s", e)
                try:
                    init_db()
                    sess = get_session()
                    vobj = create_violation(violation, session=sess)
                    amount = FINES.get(violation['violation_type'], 500.0)
                    proof = str({'snapshot': violation.get('snapshot')})
                    create_challan(vobj.id, amount=amount, proof=proof, session=sess)
                except Exception as e:
                    logger.error("Failed to save helmet violation to local DB: %s", e)

            # process vehicle detections and related violations
            for name, conf, bbox in detections:
                lname = name.lower()
                if lname not in VEHICLE_CLASSES:
                    continue

                key = create_duplicate_key(lname, bbox)
                # require consecutive detections before processing
                REQUIRED_CONSECUTIVE = 2
                count = seen_counters.get(key, 0)
                if count < REQUIRED_CONSECUTIVE:
                    seen_counters[key] = count + 1
                    continue

                # duplicate suppression
                if redis.is_duplicate(key):
                    seen_counters.pop(key, None)
                    continue

                x1, y1, x2, y2 = map(int, bbox)
                # default violation type
                violation_type = 'vehicle_detected'

                # wrong lane check (simple horizontal region)
                if detect_wrong_lane(bbox, frame.shape, forbidden_region=(0.0, 0.25)):
                    violation_type = 'wrong_lane'

                # OCR: try to aggregate from buffer or run on crop
                plate_text = None
                try:
                    candidates = ocr_buffers.get(key, [])
                    if candidates:
                        from collections import Counter
                        plate_text = Counter(candidates).most_common(1)[0][0]
                    else:
                        crop = crop_plate_from_frame(frame, (x1, y1, x2, y2))
                        if crop is not None:
                            plate_text = ocr_plate_image(crop)
                except Exception:
                    plate_text = None

                if not plate_text or len(plate_text) < 4:
                    # flag no_plate explicitly instead of silently skipping
                    if violation_type == 'vehicle_detected':
                        violation_type = 'no_plate'

                violation = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'camera_id': 'CAM_GENERIC_01',
                    'violation_type': violation_type,
                    'class': lname,
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2],
                    'ocr_text': plate_text,
                    'plate_number': plate_text,
                }

                # snapshot
                snap_dir = Path(__file__).parents[1] / 'captured_frames'
                snap_dir.mkdir(parents=True, exist_ok=True)
                snap_path = snap_dir / f"violation_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}.jpg"
                roi = frame[y1:y2, x1:x2]
                if roi is None or roi.size == 0:
                    cv2.imwrite(str(snap_path), frame)
                else:
                    cv2.imwrite(str(snap_path), roi)
                violation['snapshot'] = str(snap_path)

                # mark as seen
                redis.mark_seen(key, ttl=10)

                # persist
                try:
                    vid = firebase.save_violation(violation)
                    violation['violation_id'] = vid
                except Exception as e:
                    logger.error("Failed to save violation to firebase: %s", e)

                try:
                    init_db()
                    sess = get_session()
                    vobj = create_violation(violation, session=sess)
                    amount = FINES.get(violation['violation_type'], 500.0)
                    proof = str({'snapshot': violation.get('snapshot'), 'ocr_text': plate_text})
                    create_challan(vobj.id, amount=amount, proof=proof, session=sess)
                except Exception as e:
                    logger.error("Failed to save to local DB: %s", e)

                call_alert_stub(violation, call_log)

                # reset local buffers
                seen_counters.pop(key, None)
                ocr_buffers.pop(key, None)

            # Update OCR buffers and counters
            current_keys = set()
            for name, conf, bbox in detections:
                lname = name.lower()
                if lname in VEHICLE_CLASSES:
                    key = create_duplicate_key(lname, bbox)
                    current_keys.add(key)
                    seen_counters[key] = seen_counters.get(key, 0) + 1
                    try:
                        crop = crop_plate_from_frame(frame, bbox)
                        if crop is not None:
                            plate_text = ocr_plate_image(crop)
                            if plate_text:
                                ocr_buffers.setdefault(key, []).append(plate_text)
                    except Exception:
                        pass

            for k in list(seen_counters.keys()):
                if k not in current_keys:
                    seen_counters[k] = 0
                    ocr_buffers.pop(k, None)

            # Draw detections for visual feedback
            if detections:
                logger.debug("Detections: %s", [(d[0], round(d[1], 2), tuple(map(int, d[2]))) for d in detections])
            for det_name, det_conf, det_bbox in detections:
                dx1, dy1, dx2, dy2 = map(int, det_bbox)
                label = f"{det_name} {det_conf:.2f}"
                lname = det_name.lower()
                if lname in ('person',):
                    color = (255, 0, 0)
                elif lname in ('car', 'truck', 'bus', 'motorbike', 'motorcycle'):
                    color = (0, 255, 0)
                elif 'plate' in lname or 'number' in lname:
                    color = (0, 0, 255)
                elif 'helmet' in lname:
                    color = (255, 255, 0)
                else:
                    color = (200, 200, 200)
                cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), color, 2)
                cv2.putText(frame, label, (dx1, max(10, dy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # preview (skip in headless/test runs)
            if not nogui:
                disp = cv2.resize(frame, (800, 450)) if frame.shape[1] > 800 else frame
                cv2.imshow('Traffic Detector (press q to quit)', disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # incremental frame limit for smoke tests
            frame_counter += 0  # ensure var is referenced
            if max_frames and frame_counter >= max_frames:
                logger.info("Reached max frames (%d) - exiting", max_frames)
                break

            time.sleep(0.01)

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

