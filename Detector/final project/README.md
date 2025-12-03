Final Project: Traffic Violation Detector

This folder contains a copy of the detector scaffold.

Structure:
- traffic-violation-detector/
  - detector.py      # main detector pipeline
  - db.py            # SQLite models and helpers using SQLAlchemy
  - anpr.py          # OCR wrapper (PaddleOCR -> EasyOCR -> pytesseract)
  - services/        # firebase_service.py and redis_service.py
  - requirements.txt
  - captured_frames/  # snapshots and call_log.jsonl will be written here
- models/            # PLACE YOUR MODEL FILES HERE (see below)

Important: Model files are binary and not copied automatically. Please copy the following model files from your original repo into `final project/models/`:
- yolov8n.pt
- license_plate_detector.pt
- numberplt.pt
- helmet.pt

Recommended setup (local dev):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r "final project\traffic-violation-detector\requirements.txt"
# Optional: Install paddleocr and paddlepaddle per PaddleOCR instructions if you want best OCR accuracy
# pip install paddleocr
```

Run detector:
```powershell
cd "final project\traffic-violation-detector"
python detector.py
```

Notes:
- The detector uses an in-memory FirebaseService scaffold and a local SQLite DB (`db.sqlite`) for persistence.
- Notifications are written to `captured_frames/call_log.jsonl` as a file-based stub.
- If you want to enable a real database (Postgres) or Twilio notifications, I can help wire that up.
