import math
import cv2
import numpy as np

from ensemble_utils import (
    MethodPrediction,
    choose_digit_for_dial,
    apply_rule4,
    dial_value_from_angle,
    circular_diff_deg,
)
from classifier_infer import predict_digit_with_classifier

ZERO_ANGLE = 0.0
DIAL1_IS_CCW = True

HOUGH_DP = 1.2
HOUGH_PARAM1 = 100
HOUGH_PARAM2_LIST = (40, 30, 20)

FULL_OUTER_MASK_FRAC = 0.86
FULL_INNER_MASK_FRAC = 0.28
READ_INNER_FRAC = 0.24
READ_OUTER_FRAC = 0.72

NORMALIZED_SIZE = 160

RAY_STEP_DEG = 1.0
RAY_SAMPLES = 80
RAY_MIN_HITS = 6
RAY_MIN_RUN = 4
STRIP_HALF_WIDTH = 2
NEIGHBOR_OFFSET_DEG = 3.0

LINE_HOUGH_THRESHOLD = 14
LINE_MIN_LEN_FRAC = 0.22
LINE_MAX_GAP = 6

MIN_CONFIDENCE_WARN = 0.45
MAX_ANGLE_SPREAD_WARN = 60.0


def _avg_circles(circles, n):
    avg_x = sum(circles[0][i][0] for i in range(n)) / n
    avg_y = sum(circles[0][i][1] for i in range(n)) / n
    avg_r = sum(circles[0][i][2] for i in range(n)) / n
    return int(avg_x), int(avg_y), int(avg_r)


def _point_angle(cx, cy, px, py):
    dx = px - cx
    dy = py - cy
    return (math.degrees(math.atan2(dx, -dy)) + 360.0) % 360.0


def _draw_ticks(img, cx, cy, r, counter_clockwise=False):
    for i in range(10):
        a = math.radians(i * 36.0)
        inner = (int(cx + 0.85 * r * math.sin(a)), int(cy - 0.85 * r * math.cos(a)))
        outer = (int(cx + 1.00 * r * math.sin(a)), int(cy - 1.00 * r * math.cos(a)))
        label = (int(cx + 1.15 * r * math.sin(a)), int(cy - 1.15 * r * math.cos(a)))
        shown_digit = i if not counter_clockwise else (-i) % 10
        cv2.line(img, inner, outer, (0, 200, 0), 2)
        cv2.putText(img, str(shown_digit), label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 0), 2, cv2.LINE_AA)


def _calibrate(img_arr):
    h, w = img_arr.shape[:2]
    if h < 60 or w < 60:
        print(f"   ⚠️  Crop too small ({w}×{h}) — skipping.")
        return None

    gray = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    circles = None
    for param2 in HOUGH_PARAM2_LIST:
        circles = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, dp=HOUGH_DP,
            minDist=min(h, w) // 2,
            param1=HOUGH_PARAM1, param2=param2,
            minRadius=int(0.28 * min(h, w)),
            maxRadius=int(0.55 * min(h, w)),
        )
        if circles is not None:
            break

    if circles is None:
        print("   ❌ No circle detected.")
        return None

    return _avg_circles(circles, circles.shape[1])


def _normalize_dial(img_arr, cx, cy, r):
    pad = int(1.15 * r)
    x1 = max(0, cx - pad)
    y1 = max(0, cy - pad)
    x2 = min(img_arr.shape[1], cx + pad)
    y2 = min(img_arr.shape[0], cy + pad)

    crop = img_arr[y1:y2, x1:x2]
    if crop.size == 0:
        return img_arr, cx, cy, r

    resized = cv2.resize(crop, (NORMALIZED_SIZE, NORMALIZED_SIZE), interpolation=cv2.INTER_CUBIC)

    scale_x = NORMALIZED_SIZE / max(1, (x2 - x1))
    scale_y = NORMALIZED_SIZE / max(1, (y2 - y1))
    new_cx = int(round((cx - x1) * scale_x))
    new_cy = int(round((cy - y1) * scale_y))
    new_r = int(round(r * (scale_x + scale_y) / 2.0))

    return resized, new_cx, new_cy, new_r


def _make_full_dark_mask(gray, cx, cy, r):
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    eq = clahe.apply(blur)

    binary = cv2.adaptiveThreshold(
        eq, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 8
    )

    annulus = np.zeros_like(binary)
    cv2.circle(annulus, (cx, cy), int(FULL_OUTER_MASK_FRAC * r), 255, -1)
    cv2.circle(annulus, (cx, cy), int(FULL_INNER_MASK_FRAC * r), 0, -1)

    binary = cv2.bitwise_and(binary, annulus)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    return binary


def _make_reading_mask(full_mask, cx, cy, r):
    ring = np.zeros_like(full_mask)
    cv2.circle(ring, (cx, cy), int(READ_OUTER_FRAC * r), 255, -1)
    cv2.circle(ring, (cx, cy), int(READ_INNER_FRAC * r), 0, -1)

    reading_mask = cv2.bitwise_and(full_mask, ring)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    reading_mask = cv2.morphologyEx(reading_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    reading_mask = cv2.morphologyEx(reading_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return reading_mask


def _ray_strip_score(binary, cx, cy, ang_deg, r0, r1, strip_half_width=STRIP_HALF_WIDTH, n=RAY_SAMPLES):
    h, w = binary.shape[:2]
    a = math.radians(ang_deg)

    ux = math.sin(a)
    uy = -math.cos(a)
    px = math.cos(a)
    py = math.sin(a)

    radii = np.linspace(r0, r1, n)
    support = []
    weighted_sum = 0.0

    for rr in radii:
        vals = []
        for off in range(-strip_half_width, strip_half_width + 1):
            x = int(round(cx + rr * ux + off * px))
            y = int(round(cy + rr * uy + off * py))
            if 0 <= x < w and 0 <= y < h:
                vals.append(1 if binary[y, x] > 0 else 0)

        hit = 1 if sum(vals) >= max(1, len(vals) // 2) else 0
        support.append(hit)
        if hit:
            weighted_sum += (rr / max(r1, 1.0)) ** 1.5

    total_hits = sum(support)
    best_run = 0
    cur = 0
    for s in support:
        if s:
            cur += 1
            best_run = max(best_run, cur)
        else:
            cur = 0

    score = 1.6 * weighted_sum + 0.8 * total_hits + 2.0 * best_run
    return float(score), int(total_hits), int(best_run)


def _ray_angle(reading_mask, cx, cy, r):
    best_ang = None
    best_score = -1e9
    r0 = READ_INNER_FRAC * r
    r1 = READ_OUTER_FRAC * r

    for ang in np.arange(0.0, 360.0, RAY_STEP_DEG):
        score, hits, run = _ray_strip_score(reading_mask, cx, cy, ang, r0, r1)
        if hits < RAY_MIN_HITS or run < RAY_MIN_RUN:
            continue

        s1, _, _ = _ray_strip_score(reading_mask, cx, cy, (ang - NEIGHBOR_OFFSET_DEG) % 360.0, r0, r1, strip_half_width=1, n=60)
        s2, _, _ = _ray_strip_score(reading_mask, cx, cy, (ang + NEIGHBOR_OFFSET_DEG) % 360.0, r0, r1, strip_half_width=1, n=60)

        final_score = score + 0.20 * (s1 + s2)
        if final_score > best_score:
            best_score = final_score
            best_ang = float(ang)

    return best_ang


def _contour_angle(reading_mask, cx, cy, r):
    contours, _ = cv2.findContours(reading_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_ang = None
    best_score = -1e9

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 8:
            continue

        pts = cnt[:, 0, :].astype(np.float32)
        dx = pts[:, 0] - cx
        dy = pts[:, 1] - cy
        dists = np.sqrt(dx**2 + dy**2)

        keep = (dists > 0.30 * r) & (dists < 0.78 * r)
        outer_pts = pts[keep]
        if len(outer_pts) < 5:
            continue

        vx = outer_pts[:, 0] - cx
        vy = outer_pts[:, 1] - cy
        norms = np.sqrt(vx**2 + vy**2) + 1e-6
        ux = vx / norms
        uy = vy / norms

        mean_x = float(np.mean(ux))
        mean_y = float(np.mean(uy))
        strength = float(np.hypot(mean_x, mean_y))
        if strength < 0.15:
            continue

        tip_x = cx + mean_x * 0.60 * r
        tip_y = cy + mean_y * 0.60 * r
        ang = _point_angle(cx, cy, tip_x, tip_y)

        score = 3.0 * strength + 0.005 * area
        if score > best_score:
            best_score = score
            best_ang = float(ang)

    return best_ang


def _line_centre_dist(cx, cy, x1, y1, x2, y2):
    num = abs((y2 - y1) * cx - (x2 - x1) * cy + x2 * y1 - y2 * x1)
    den = math.hypot(y2 - y1, x2 - x1) + 1e-6
    return num / den


def _line_angle(reading_mask, cx, cy, r):
    lines = cv2.HoughLinesP(
        reading_mask,
        rho=1,
        theta=np.pi / 180,
        threshold=LINE_HOUGH_THRESHOLD,
        minLineLength=int(LINE_MIN_LEN_FRAC * r),
        maxLineGap=LINE_MAX_GAP,
    )
    if lines is None:
        return None

    best_ang = None
    best_score = -1e9

    for line in lines:
        x1, y1, x2, y2 = line[0]
        d1 = float(np.hypot(x1 - cx, y1 - cy))
        d2 = float(np.hypot(x2 - cx, y2 - cy))
        near = min(d1, d2)
        far = max(d1, d2)

        if near > 0.40 * r:
            continue
        if far < 0.35 * r or far > 0.80 * r:
            continue

        lc = _line_centre_dist(cx, cy, x1, y1, x2, y2)
        if lc > 0.18 * r:
            continue

        tip = (x1, y1) if d1 > d2 else (x2, y2)
        ang = _point_angle(cx, cy, tip[0], tip[1])

        length = float(np.hypot(x2 - x1, y2 - y1))
        score = length - 3.0 * lc

        if score > best_score:
            best_score = score
            best_ang = float(ang)

    return best_ang


def _dial_zero_angle(norm_img, cx, cy, r, fallback=ZERO_ANGLE):
    _ = (norm_img, cx, cy, r)
    return float(fallback)


def _angle_spread_deg(angle_list):
    if len(angle_list) < 2:
        return 0.0
    max_sep = 0.0
    for i in range(len(angle_list)):
        for j in range(i + 1, len(angle_list)):
            max_sep = max(max_sep, circular_diff_deg(angle_list[i], angle_list[j]))
    return float(max_sep)


def _best_pointer_angle(ray_ang, contour_ang, line_ang):
    if ray_ang is not None:
        return float(ray_ang), "ray"
    if contour_ang is not None:
        return float(contour_ang), "contour"
    if line_ang is not None:
        return float(line_ang), "line"
    return 0.0, "fallback"


def _neighbor_digits(value):
    lower_digit = int(math.floor(value + 1e-6)) % 10
    upper_digit = (lower_digit + 1) % 10
    return lower_digit, upper_digit


def _run_dial(img_arr, counter_clockwise, gauge_index, dial_name, debug):
    calib = _calibrate(img_arr)
    if calib is None:
        if debug:
            cv2.imwrite(f"failed_crop_{gauge_index}.png", img_arr)
        return 0, 0.0, 0.0, 0.0

    cx, cy, r = calib
    print(f"   ⭕ Circle: centre=({cx},{cy})  r={r}px")

    norm_img, cx, cy, r = _normalize_dial(img_arr, cx, cy, r)
    gray = cv2.cvtColor(norm_img, cv2.COLOR_BGR2GRAY)

    full_mask = _make_full_dark_mask(gray, cx, cy, r)
    reading_mask = _make_reading_mask(full_mask, cx, cy, r)

    ray_ang = _ray_angle(reading_mask, cx, cy, r)
    contour_ang = _contour_angle(reading_mask, cx, cy, r)
    line_ang = _line_angle(reading_mask, cx, cy, r)
    clf_digit, clf_conf = predict_digit_with_classifier(norm_img, dial_name)

    print(
        f"   📎 Candidates: "
        f"ray={f'{ray_ang:.1f}°' if ray_ang is not None else 'None'}  "
        f"contour={f'{contour_ang:.1f}°' if contour_ang is not None else 'None'}  "
        f"line={f'{line_ang:.1f}°' if line_ang is not None else 'None'}  "
        f"classifier={clf_digit if clf_digit is not None else 'None'}"
    )

    zero_angle = _dial_zero_angle(norm_img, cx, cy, r, fallback=ZERO_ANGLE)

    predictions = [
        MethodPrediction("ray", ray_ang, 0.72 if ray_ang is not None else 0.0),
        MethodPrediction("contour", contour_ang, 0.62 if contour_ang is not None else 0.0),
        MethodPrediction("line", line_ang, 0.55 if line_ang is not None else 0.0),
    ]
    if clf_digit is not None:
        predictions.append(MethodPrediction("classifier", None, clf_conf, digit=clf_digit))

    ensemble_digit, conf, score_by_digit, usable = choose_digit_for_dial(
        predictions=predictions,
        counter_clockwise=counter_clockwise,
        zero_angle=zero_angle,
    )

    angle_preds = [p for p in usable if p.abs_angle is not None]
    angle_spread = _angle_spread_deg([p.abs_angle for p in angle_preds])

    chosen_angle, angle_source = _best_pointer_angle(ray_ang, contour_ang, line_ang)

    value, lower_digit, frac, rel_ang_cw = dial_value_from_angle(
        abs_angle=chosen_angle,
        counter_clockwise=counter_clockwise,
        zero_angle=zero_angle,
    )

    print(f"   🗳  Vote scores: {score_by_digit}")
    print(f"   🎯 Pointer source={angle_source}  ensemble digit={ensemble_digit}  conf={conf:.2f}")
    print(f"   🔄 Dial direction: {'CCW ↺' if counter_clockwise else 'CW ↻'}")
    print(f"   📍 Zero angle={zero_angle:.1f}°")
    print(f"   📐 Unified CW angle={rel_ang_cw:.1f}°")
    print(f"   🔢 Continuous value={value:.3f}  frac={frac:.3f}  -> lower digit={lower_digit}")
    print(f"   🧭 Reading from red-dot angle={chosen_angle:.1f}°")

    if conf < MIN_CONFIDENCE_WARN or angle_spread > MAX_ANGLE_SPREAD_WARN:
        print(
            f"   ⚠️  Low-confidence dial read: conf={conf:.2f}, "
            f"angle spread={angle_spread:.1f}°"
        )

    if debug:
        tip_x = int(round(cx + 0.65 * r * math.sin(math.radians(chosen_angle))))
        tip_y = int(round(cy - 0.65 * r * math.cos(math.radians(chosen_angle))))
        vis = norm_img.copy()
        cv2.circle(vis, (cx, cy), r, (0, 0, 220), 2)
        _draw_ticks(vis, cx, cy, r, counter_clockwise=counter_clockwise)
        cv2.arrowedLine(vis, (cx, cy), (tip_x, tip_y), (220, 0, 0), 2, tipLength=0.15)
        cv2.circle(vis, (tip_x, tip_y), 5, (0, 0, 220), -1)
        cv2.putText(vis, f"conf={conf:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 160, 0), 2)
        lower_dbg, upper_dbg = _neighbor_digits(value)
        cv2.putText(vis, f"val={value:.2f}", (10, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 160, 0), 2)
        cv2.putText(vis, f"read={lower_dbg} (next {upper_dbg})", (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 160, 0), 2)

        board = np.hstack((cv2.cvtColor(reading_mask, cv2.COLOR_GRAY2BGR), vis))
        path = f"storyboard_dial_{gauge_index}.png"
        cv2.imwrite(path, board)
        print(f"   Saved: {path}")

    return int(lower_digit), float(value), float(rel_ang_cw), float(conf)


def _place_name_from_index(index, n_dials):
    power = n_dials - 1 - index
    return f"{1000 * (10 ** power):,}"


def r_main(cropped_gauges, debug=False):
    if not cropped_gauges:
        print("⚠️  No gauge crops to read.")
        return 0, []

    dial_names = [f"dial_{i+1}" for i in range(len(cropped_gauges))]

    print(f"\n{'═' * 60}")
    print(f"  Reading {len(cropped_gauges)} dial(s)  [D1 bottom, then top left -> right]")
    print(f"{'═' * 60}")

    digits = []
    values = []
    rel_angles = []
    confidences = []

    for i, crop in enumerate(cropped_gauges):
        ccw = (i % 2 == 0) if DIAL1_IS_CCW else (i % 2 == 1)
        place_name = _place_name_from_index(i, len(cropped_gauges))

        print(
            f"\n  ── Dial {i+1}  [{place_name:>12} place]  "
            f"({'CCW ↺' if ccw else 'CW ↻'})  "
            f"{crop.shape[1]}×{crop.shape[0]} px"
        )

        d, v, a, c = _run_dial(
            crop,
            counter_clockwise=ccw,
            gauge_index=i + 1,
            dial_name=dial_names[i],
            debug=debug,
        )

        digits.append(d)
        values.append(v)
        rel_angles.append(a)
        confidences.append(c)

    print(f"\n  Raw digits  (in order) : {digits}")
    print(f"  Values      (in order) : {[f'{v:.3f}' for v in values]}")
    print(f"  Rel angles  (in order) : {[f'{a:.1f}°' for a in rel_angles]}")
    print(f"  Confidence  (in order) : {[f'{c:.2f}' for c in confidences]}")

    print("\n  Applying Rule 4 ...")
    digits_rtl = list(reversed(digits))
    values_rtl = list(reversed(values))
    corrected_rtl = apply_rule4(digits_rtl, values_rtl, on_digit_tol=0.08)
    corrected = list(reversed(corrected_rtl))
    print(f"  Corrected   (in order) : {corrected}")

    number_str = "".join(str(d) for d in corrected)
    reading = int(number_str) * 1000

    print(f"\n{'═' * 60}")
    print(f"  Dials (in order)  : {corrected}")
    print(f"  Meter reading     : {number_str},000 cubic feet")
    print(f"  Numeric value     : {reading:,}")
    print(f"{'═' * 60}\n")

    return int(reading), corrected
