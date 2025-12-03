from pathlib import Path
import re
import cv2
import numpy as np
from typing import Optional

_paddleocr = None
_easyocr = None
_pytesseract = None


def _normalize(text: str) -> str:
    if not text:
        return ''
    # keep alphanumeric and uppercase
    t = re.sub(r'[^A-Za-z0-9]', '', text)
    return t.upper()

from pathlib import Path
import re
import cv2
import numpy as np
from typing import Optional
from collections import Counter

_paddleocr = None
_easyocr = None
_pytesseract = None


def _normalize(text: str) -> str:
    if not text:
        return ''
    # keep alphanumeric and uppercase
    t = re.sub(r'[^A-Za-z0-9]', '', text)
    return t.upper()


def _is_plausible_plate(s: str) -> bool:
    if not s:
        return False
    s = _normalize(s)
    if len(s) < 6 or len(s) > 12:
        return False
    letters = sum(1 for c in s if c.isalpha())
    digits = sum(1 for c in s if c.isdigit())
    return letters >= 2 and digits >= 2


def _preprocess_plate(img: np.ndarray) -> np.ndarray:
    # convert to gray and resize to a reasonable width
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    target_w = 400
    if w < target_w:
        scale = target_w / float(w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    # improve contrast
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    except Exception:
        pass

    # denoise
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # adaptive threshold
    try:
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    except Exception:
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # morphological close to join characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)


def _try_paddle(image) -> Optional[str]:
    global _paddleocr
    try:
        if _paddleocr is None:
            from paddleocr import PaddleOCR
            _paddleocr = PaddleOCR(use_angle_cls=False, lang='en')
        res = _paddleocr.ocr(image, cls=False)
        texts = []
        for line in res:
            if len(line) >= 2 and isinstance(line[1], (list, tuple)):
                text = line[1][0]
            elif len(line) >= 2 and isinstance(line[1], dict):
                text = line[1].get('text', '')
            else:
                text = ''
            texts.append(text)
        joined = ' '.join(texts)
        return _normalize(joined)
    except Exception:
        return None


def _try_easyocr(image) -> Optional[str]:
    global _easyocr
    try:
        if _easyocr is None:
            import easyocr
            _easyocr = easyocr.Reader(['en'], gpu=False)
        res = _easyocr.readtext(image)
        texts = [t[1] for t in res]
        return _normalize(' '.join(texts))
    except Exception:
        return None


def _try_pytesseract(image) -> Optional[str]:
    global _pytesseract
    try:
        import pytesseract
        _pytesseract = pytesseract
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(th, config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        return _normalize(text)
    except Exception:
        return None


def ocr_plate_image(image) -> Optional[str]:
    """Preprocess plate image, run multiple OCRs and select the best candidate.

    Returns normalized alphanumeric string or None.
    """
    if image is None:
        return None

    h, w = image.shape[:2]
    if h == 0 or w == 0:
        return None

    pre = _preprocess_plate(image)

    results = []
    # try multiple engines and collect candidates
    for fn in (_try_paddle, _try_easyocr, _try_pytesseract):
        try:
            t = fn(pre)
            if t:
                results.append(t)
        except Exception:
            continue

    if not results:
        # fallback: try without heavy preprocessing
        try:
            t = _try_pytesseract(image)
            if t:
                results.append(t)
        except Exception:
            pass

    if not results:
        return None

    # majority vote
    c = Counter(results)
    candidate, count = c.most_common(1)[0]

    # prefer plausible plate if available
    if _is_plausible_plate(candidate):
        return candidate
    for r in results:
        if _is_plausible_plate(r):
            return r

    # otherwise return the most common candidate
    return candidate


def crop_plate_from_frame(frame, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    # resize small crops to help OCR
    if crop.shape[0] < 40:
        crop = cv2.resize(crop, (crop.shape[1] * 2, crop.shape[0] * 2), interpolation=cv2.INTER_LINEAR)
    return crop
