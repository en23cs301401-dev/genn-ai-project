"""Pure helper functions for Food Verse AI (no Streamlit / OpenAI imports, so they are easy to test)."""
import base64
import io
import json
import os
import re
import tempfile

import cv2
import pandas as pd
from PIL import Image, ImageOps

NUTRI_COLS = ["calories", "protein_g", "carbs_g", "fat_g"]

ACTIVITY_FACTORS = {
    "Sedentary": 1.2,
    "Light": 1.375,
    "Moderate": 1.55,
    "Very active": 1.725,
}


def safe_int(value, default=0, lo=0, hi=100):
    """Convert model output to an int inside [lo, hi]; fall back to default."""
    try:
        return max(lo, min(hi, int(round(float(value)))))
    except (TypeError, ValueError):
        return default


def parse_json(text):
    """Parse JSON from a model reply, tolerating markdown code fences."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    return json.loads(text)


def rows_to_df(rows):
    """Build a DataFrame from model rows and guarantee numeric nutrition columns."""
    if not isinstance(rows, list):
        rows = []
    rows = [r for r in rows if isinstance(r, dict)]
    df = pd.DataFrame(rows)
    for col in NUTRI_COLS:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def totals(df):
    return {c: float(df[c].sum()) for c in NUTRI_COLS}


def nutrition_targets(age, sex, height_cm, weight_kg, activity, goal):
    """BMI, BMR (Mifflin-St Jeor), TDEE, calorie target and macro targets."""
    bmr = 10 * weight_kg + 6.25 * height_cm - 5 * age + (5 if sex == "Male" else -161)
    tdee = bmr * ACTIVITY_FACTORS[activity]
    delta = {"Weight loss": -500, "Maintain weight": 0, "Muscle gain": 300}[goal]
    floor = 1500 if sex == "Male" else 1200  # never recommend below a safe minimum
    target = max(tdee + delta, floor)
    protein_per_kg = {"Weight loss": 1.6, "Maintain weight": 1.4, "Muscle gain": 1.8}[goal]
    protein = weight_kg * protein_per_kg
    fat = target * 0.25 / 9
    carbs = max((target - protein * 4 - fat * 9) / 4, 0)
    bmi = weight_kg / ((height_cm / 100) ** 2)
    return {
        "BMI": round(bmi, 1),
        "BMR": round(bmr),
        "TDEE": round(tdee),
        "Target kcal": round(target),
        "Protein g": round(protein),
        "Carbs g": round(carbs),
        "Fat g": round(fat),
    }


def prepare_image(data, max_side=1024):
    """Fix rotation, convert to RGB, downscale and re-encode as JPEG (saves tokens and cost)."""
    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def to_data_url(jpeg_bytes, mime="image/jpeg"):
    return f"data:{mime};base64,{base64.b64encode(jpeg_bytes).decode()}"


def extract_frames(video_bytes, n=4, max_width=768):
    """Return up to n evenly spaced frames from a video as JPEG bytes."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        path = tmp.name

    frames = []
    cap = cv2.VideoCapture(path)
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        raw = []
        if total > 0:
            for i in range(n):
                cap.set(cv2.CAP_PROP_POS_FRAMES, total * (2 * i + 1) // (2 * n))
                ok, frame = cap.read()
                if ok:
                    raw.append(frame)
        else:  # some containers do not report a frame count: read sequentially
            allf = []
            while len(allf) < 900:
                ok, frame = cap.read()
                if not ok:
                    break
                allf.append(frame)
            if allf:
                step = max(len(allf) // n, 1)
                raw = allf[step // 2::step][:n]

        for frame in raw:
            h, w = frame.shape[:2]
            if w > max_width:
                frame = cv2.resize(frame, (max_width, int(h * max_width / w)))
            ok, buf = cv2.imencode(".jpg", frame)
            if ok:
                frames.append(buf.tobytes())
    finally:
        cap.release()
        try:
            os.remove(path)
        except OSError:
            pass
    return frames
