import argparse
import os
import time
import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

from services.firebase_service import FirebaseService
from services.redis_service import RedisService


def load_models(models_dir):
    models = []
    if YOLO is None:
        print("ultralytics not available; running without ML models")
        return models
    for p in Path(models_dir).glob("*.pt"):
        try:
            print(f"Loading model from {p}")
            models.append((p.stem, YOLO(str(p))))
        except Exception as e:
            print(f"Failed to load {p}: {e}")
    return models


def annotate_frame(frame, detections):
    # detections: list of dicts {label, conf, xyxy}
    for det in detections:
        x1, y1, x2, y2 = map(int, det['xyxy'])
        label = det['label']
        conf = det['conf']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        txt = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 6, y1), (0, 255, 0), -1)
        cv2.putText(frame, txt, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return frame


def run(video_source, out_dir, no_gui=False, max_frames=None):
    models_dir = Path(__file__).resolve().parents[1] / 'models'
    models = load_models(models_dir)

    fb = FirebaseService()
    cache = RedisService()

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(out_dir) / 'call_log.jsonl'

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Failed to open video source: {video_source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video_path = Path(out_dir) / f"annotated_{int(time.time())}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (w, h))

    frame_idx = 0
    saved_snapshots = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break

            all_detections = []

            for name, model in models:
                try:
                    results = model(frame)
                except Exception as e:
                    print(f"Model {name} inference failed: {e}")
                    continue
                # results can be a list-like; take first
                if len(results) == 0:
                    continue
                r = results[0]
                boxes = getattr(r, 'boxes', None)
                if boxes is None:
                    continue
                try:
                    xyxy = boxes.xyxy.cpu().numpy()
                    confs = boxes.conf.cpu().numpy()
                    classes = boxes.cls.cpu().numpy().astype(int)
                except Exception:
                    # fallback if tensors not present
                    try:
                        xyxy = np.array(boxes.xyxy)
                        confs = np.array(boxes.conf)
                        classes = np.array(boxes.cls).astype(int)
                    except Exception:
                        continue

                for b, c, cf in zip(xyxy, classes, confs):
                    lbl = f"{name}:{c}"
                    det = {'label': lbl, 'conf': float(cf), 'xyxy': b.tolist()}
                    all_detections.append(det)

            if all_detections:
                annotate_frame(frame, all_detections)
                # Save snapshot every detection (rate-limited to avoid flooding)
                if frame_idx % 10 == 0:
                    snap_path = Path(out_dir) / f"snap_{frame_idx}.jpg"
                    cv2.imwrite(str(snap_path), frame)
                    saved_snapshots += 1

                # Log to jsonl
                entry = {
                    'timestamp': datetime.utcnow().isoformat() + 'Z',
                    'frame': frame_idx,
                    'detections': [{'label': d['label'], 'conf': d['conf'], 'xyxy': d['xyxy']} for d in all_detections],
                    'snapshot': str(snap_path) if saved_snapshots else None
                }
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(entry) + "\n")

            writer.write(frame)

            if not no_gui:
                cv2.imshow('annotated', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        cap.release()
        writer.release()
        if not no_gui:
            cv2.destroyAllWindows()

    print(f"Done. Annotated video saved to: {out_video_path}")
    print(f"Snapshots saved to: {out_dir}")
    print(f"Call log: {log_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-v', '--video', help='Path to input video file. If omitted, webcam is used.', default=None)
    p.add_argument('-o', '--outdir', help='Output directory to save annotated video and snapshots', required=True)
    p.add_argument('--nogui', action='store_true', help='Run without showing GUI')
    p.add_argument('--max-frames', type=int, default=None, help='Stop after this many frames')
    args = p.parse_args()

    source = args.video if args.video else 0
    run(source, args.outdir, no_gui=args.nogui, max_frames=args.max_frames)


if __name__ == '__main__':
    main()
import argparse
import os
import time
import json
import logging
from pathlib import Path
from datetime import datetime

import cv2
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

from detector import detect_no_helmet, detect_wrong_lane, create_duplicate_key, _bbox_center, _iou  # reuse helpers
from services.firebase_service import FirebaseService
from services.redis_service import RedisService
from anpr import ocr_plate_image, crop_plate_from_frame

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorbike', 'motorcycle'}
FINES = {'no_helmet': 500.0, 'vehicle_detected': 1000.0, 'no_plate': 1500.0, 'wrong_lane': 700.0}


def call_alert_stub(violation: dict, outpath: Path):
    entry = violation.copy()
    entry['alerted_at'] = datetime.utcnow().isoformat()
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open('a', encoding='utf8') as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info("Call/alert logged to %s", outpath)


def load_models(models_dir: Path):
    models = []
    if YOLO is None:
        logger.warning("ultralytics YOLO not available; running without ML.")
        return models
    if not models_dir.exists():
        logger.warning("Models dir not found: %s", models_dir)
        return models
    for p in sorted(models_dir.glob('*.pt')):
        try:
            logger.info("Loading model %s", p.name)
            models.append((p.name, YOLO(str(p))))
        except Exception as e:
            logger.error("Failed to load %s: %s", p, e)
    return models


def annotate_and_save_frame(frame, detections, out_video_writer):
    # draw boxes + labels then write frame to video writer
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
    out_video_writer.write(frame)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-v', '--video', required=True, help='Path to input video file (or device index like 0)')
    p.add_argument('-o', '--output', required=True, help='Output folder to save annotated video + snapshots')
    p.add_argument('--disable-model', action='store_true', help='Skip loading ML models')
    p.add_argument('--nogui', action='store_true', help='Do not show preview window')
    p.add_argument('--max-frames', type=int, default=0, help='Stop after N frames (0 = endless)')
    args = p.parse_args()

    video_src = args.video
    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    call_log = out_dir / 'call_log.jsonl'
    annotated_path = out_dir / f"annotated_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.mp4"

    firebase = FirebaseService()  # uses in-memory if no creds
    redis = RedisService()

    # load models
    models_dir = Path(__file__).parents[1] / 'models'
    models_list = [] if args.disable_model else load_models(models_dir)

    # open capture
    try:
        cap = cv2.VideoCapture(int(video_src)) if str(video_src).isdigit() else cv2.VideoCapture(str(video_src))
    except Exception:
        cap = cv2.VideoCapture(str(video_src))

    if not cap.isOpened():
        logger.error("Failed to open video source: %s", video_src)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(str(annotated_path), fourcc, fps, (w, h))

    logger.info("Writing annotated video to %s", annotated_path)

    frame_counter = 0
    seen_counters = {}
    ocr_buffers = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of input stream")
                break
            frame_counter += 1
            detections = []

            if models_list:
                for mname, mobj in models_list:
                    try:
                        results = mobj(frame, conf=0.25, verbose=False)
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

            # helmet detection (and save snapshot + call log)
            helmet_issues = detect_no_helmet(detections)
            for bike_bbox, person_bbox in helmet_issues:
                key = create_duplicate_key('no_helmet', bike_bbox)
                if redis.is_duplicate(key):
                    continue
                redis.mark_seen(key, ttl=10)
                x1, y1, x2, y2 = map(int, bike_bbox)
                violation = {'timestamp': datetime.utcnow().isoformat(), 'violation_type': 'no_helmet', 'bbox': [x1, y1, x2, y2]}
                snap_path = out_dir / f"violation_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}.jpg"
                roi = frame[y1:y2, x1:x2]
                cv2.imwrite(str(snap_path), roi if roi is not None and roi.size else frame)
                violation['snapshot'] = str(snap_path)
                call_alert_stub(violation, call_log)

            # vehicle processing (OCR + save snapshot)
            for name, conf, bbox in detections:
                lname = name.lower()
                if lname not in VEHICLE_CLASSES:
                    continue
                key = create_duplicate_key(lname, bbox)
                REQUIRED_CONSECUTIVE = 2
                count = seen_counters.get(key, 0)
                if count < REQUIRED_CONSECUTIVE:
                    seen_counters[key] = count + 1
                    continue
                if redis.is_duplicate(key):
                    seen_counters.pop(key, None)
                    continue

                x1, y1, x2, y2 = map(int, bbox)
                violation_type = 'vehicle_detected'
                if detect_wrong_lane(bbox, frame.shape, forbidden_region=(0.0, 0.25)):
                    violation_type = 'wrong_lane'

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
                    if violation_type == 'vehicle_detected':
                        violation_type = 'no_plate'

                violation = {'timestamp': datetime.utcnow().isoformat(), 'violation_type': violation_type, 'class': lname, 'confidence': conf, 'bbox': [x1, y1, x2, y2], 'plate_number': plate_text}
                snap_path = out_dir / f"violation_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}.jpg"
                roi = frame[y1:y2, x1:x2]
                cv2.imwrite(str(snap_path), roi if roi is not None and roi.size else frame)
                violation['snapshot'] = str(snap_path)
                redis.mark_seen(key, ttl=10)
                call_alert_stub(violation, call_log)
                seen_counters.pop(key, None)
                ocr_buffers.pop(key, None)

            # update OCR buffers and counters
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

            # annotate and write frame
            annotate_and_save_frame(frame, detections, out_writer)

            if not args.nogui:
                disp = cv2.resize(frame, (800, 450)) if frame.shape[1] > 800 else frame
                cv2.imshow('Annotated', disp)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if args.max_frames and frame_counter >= args.max_frames:
                logger.info("Reached max frames (%d) - exiting", args.max_frames)
                break

            time.sleep(0.01)

    finally:
        cap.release()
        out_writer.release()
        cv2.destroyAllWindows()
        logger.info("Done. Annotated video saved to %s", annotated_path)


if __name__ == '__main__':
    main()