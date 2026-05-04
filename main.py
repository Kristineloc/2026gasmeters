"""
main.py
=======
End-to-end gas meter pipeline.

Reading order in this version:
- Dial 1 = bottom-left dial
- Dial 2..Dial N = top-row dials from left to right

For a 5-dial meter:
    D1 = bottom-left
    D2 = top-left
    D3 = top-mid-left
    D4 = top-mid-right
    D5 = top-right
"""

from pathlib import Path
from time import sleep
import datetime

from detection_main import d_main
from reading_main import r_main

PERIOD = 15
DEBUG = True
USE_CAMERA = False
SAMPLE_IMAGE = "meter9.jpg"
EXPECTED_GAUGES = 5


BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = BASE_DIR / "GM_sample_images"
CAPTURE_DIR = BASE_DIR / "GM_captured_images"
CAPTURE_DIR.mkdir(exist_ok=True)


def _digits_to_reading(digits_in_order):
    return int("".join(str(d) for d in digits_in_order)) * 1000


def _place_name_from_index(index, n_dials):
    power = n_dials - 1 - index
    return f"{1000 * (10 ** power):,}"


def reading_loop():
    try:
        if USE_CAMERA:
            from RP_Camera.capture import CameraController
            camera = CameraController(save_dir=str(CAPTURE_DIR))
            filename, capture_time = camera.capture()
            camera.stop()
            image_path = str(CAPTURE_DIR / filename)
            image_name = Path(filename).name
        else:
            capture_time = datetime.datetime.now()
            image_path = str(SAMPLE_DIR / SAMPLE_IMAGE)
            image_name = SAMPLE_IMAGE

        print(f"\n{'─' * 60}")
        print(f"  Image     : {image_path}")
        print(f"  Timestamp : {capture_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'─' * 60}")

        cropped_gauges = d_main(
            image_path=image_path,
            expected_gauges=EXPECTED_GAUGES,
            debug=DEBUG,
        )

        if not cropped_gauges:
            print("⚠️  No gauges detected — skipping this cycle.")
            return

        print(f"✅  Cropped {len(cropped_gauges)} dial(s).")

        final_reading, final_digits = r_main(cropped_gauges, debug=DEBUG)



        print("\n" + "━" * 62)
        print("  DIAL-BY-DIAL SUMMARY")
        print("  (Dial 1 = bottom-left, then top row left -> right)")
        print("━" * 62)
        for i in range(len(cropped_gauges)):
            place_name = _place_name_from_index(i, len(cropped_gauges))
            debug_file = f"debug_crop_{i+1}.png" if DEBUG else "—"
            print(f"  Dial {i+1}  [{place_name:>12} place]  →  {debug_file}")
        print("━" * 62)
        print(f"  DIGITS IN ORDER : {final_digits}")
        print(f"  TIMESTAMP       : {capture_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  FINAL READING   : {final_reading:,} cubic feet")
        print("━" * 62 + "\n")

    except FileNotFoundError as exc:
        print(f"❌  Image not found: {exc}")
    except Exception as exc:
        print(f"❌  Unexpected error: {exc}")
        raise


if __name__ == "__main__":
    print("Gas Meter Reader started. Press Ctrl-C to stop.")
    while True:
        reading_loop()
        sleep(PERIOD)
        break
