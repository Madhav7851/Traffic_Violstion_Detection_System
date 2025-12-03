"""Lightweight smoke test: simulate a violation and exercise services/DB without camera or ML.

Run with the project venv to verify core persistence and scaffold pieces work.
"""
from pathlib import Path
from datetime import datetime

from services.firebase_service import FirebaseService
from services.redis_service import RedisService
from db import init_db, get_session, create_violation, create_challan


def main():
    base = Path(__file__).parent
    # init services
    firebase = FirebaseService()
    redis = RedisService()

    # simple sample violation
    violation = {
        'timestamp': datetime.utcnow().isoformat(),
        'camera_id': 'SMOKE_CAM',
        'violation_type': 'no_plate',
        'class': 'car',
        'confidence': 0.0,
        'bbox': [0, 0, 10, 10],
        'ocr_text': None,
        'snapshot': str((base / 'captured_frames' / 'smoke.jpg').resolve())
    }

    # ensure captured_frames exists and write a placeholder image file
    cap_dir = base / 'captured_frames'
    cap_dir.mkdir(parents=True, exist_ok=True)
    (cap_dir / 'smoke.jpg').write_text('smoke placeholder')

    # save to firebase scaffold
    vid = firebase.save_violation(violation)
    print('Firebase saved id:', vid)

    # persist to local sqlite
    init_db()
    sess = get_session()
    vobj = create_violation(violation, session=sess)
    print('DB violation id:', vobj.id)
    c = create_challan(vobj.id, amount=100.0, proof='smoke', session=sess)
    print('Challan id:', c.id)

    # test redis duplicate logic
    k = 'SMOKE:1-2-3-4'
    print('redis.is_duplicate before:', redis.is_duplicate(k))
    redis.mark_seen(k, ttl=2)
    print('redis.is_duplicate after mark:', redis.is_duplicate(k))


if __name__ == '__main__':
    main()
