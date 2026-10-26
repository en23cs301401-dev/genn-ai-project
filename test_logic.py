"""Run with:  python tests/test_logic.py   (or: pytest)"""
import io
import os
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logic  # noqa: E402


def test_safe_int():
    assert logic.safe_int("85") == 85
    assert logic.safe_int(72.6) == 73
    assert logic.safe_int(None) == 0
    assert logic.safe_int("abc", default=50) == 50
    assert logic.safe_int(250) == 100
    assert logic.safe_int(-5) == 0


def test_parse_json_with_fences():
    assert logic.parse_json('{"a": 1}') == {"a": 1}
    assert logic.parse_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_rows_to_df_handles_missing_and_bad_values():
    df = logic.rows_to_df([{"name": "rice", "calories": "200"}, {"name": "dal", "protein_g": None}])
    assert list(df["calories"]) == [200, 0]
    t = logic.totals(df)
    assert t["calories"] == 200 and t["fat_g"] == 0
    assert logic.totals(logic.rows_to_df(None))["calories"] == 0
    assert logic.totals(logic.rows_to_df(["junk", 5]))["calories"] == 0


def test_nutrition_targets():
    t = logic.nutrition_targets(22, "Male", 170, 65, "Moderate", "Maintain weight")
    # BMR = 650 + 1062.5 - 110 + 5 = 1607.5 ; TDEE = 1607.5 * 1.55
    assert t["BMR"] == round(1607.5)
    assert t["TDEE"] == round(1607.5 * 1.55)
    assert t["Target kcal"] == t["TDEE"]
    loss = logic.nutrition_targets(22, "Male", 170, 65, "Moderate", "Weight loss")
    gain = logic.nutrition_targets(22, "Male", 170, 65, "Moderate", "Muscle gain")
    assert loss["Target kcal"] < t["Target kcal"] < gain["Target kcal"]
    # safety floor
    tiny = logic.nutrition_targets(60, "Female", 140, 40, "Sedentary", "Weight loss")
    assert tiny["Target kcal"] >= 1200
    assert t["BMI"] == round(65 / 1.7 ** 2, 1)


def _png_bytes(size=(2000, 1000), mode="RGBA"):
    img = Image.new(mode, size, (200, 30, 30, 128) if mode == "RGBA" else "red")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def test_prepare_image_resizes_and_converts():
    out = logic.prepare_image(_png_bytes())
    img = Image.open(io.BytesIO(out))
    assert img.format == "JPEG" and max(img.size) <= 1024
    assert logic.to_data_url(out).startswith("data:image/jpeg;base64,")


def test_prepare_image_rejects_garbage():
    try:
        logic.prepare_image(b"not an image")
    except Exception:
        return
    raise AssertionError("expected an error for invalid image data")


def test_extract_frames_from_video():
    path = os.path.join(tempfile.gettempdir(), "fv_test.mp4")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (320, 240))
    for i in range(60):
        frame = np.full((240, 320, 3), i * 4 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    with open(path, "rb") as f:
        frames = logic.extract_frames(f.read(), n=4)
    os.remove(path)
    assert len(frames) == 4
    assert all(Image.open(io.BytesIO(fr)).size == (320, 240) for fr in frames)


def test_extract_frames_bad_video_returns_empty():
    assert logic.extract_frames(b"this is not a video", n=4) == []


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"\nAll {len(tests)} tests passed")
