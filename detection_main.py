"""
detection_main.py
=================
Use Roboflow detections to crop the meter dials.

Returned order in this version:
- Dial 1 = bottom-left dial
- Dial 2..Dial N = top-row dials from left to right

For a 5-dial meter:
    D1 = bottom-left
    D2 = top-left
    D3 = top-mid-left
    D4 = top-mid-right
    D5 = top-right
"""

import cv2
from inference_sdk import InferenceHTTPClient
from secret import ROBOFLOW_API_KEY, WORKSPACE_NAME, WORKFLOW_ID

_client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=ROBOFLOW_API_KEY,
)


def _find_predictions(obj):
    if isinstance(obj, dict):
        if "predictions" in obj and isinstance(obj["predictions"], list):
            return obj["predictions"]
        for value in obj.values():
            found = _find_predictions(value)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _find_predictions(item)
            if found is not None:
                return found
    return None


def _print_predictions(predictions):
    for p in predictions:
        print(
            f"  class={p.get('class', '?'):10s}  "
            f"conf={p.get('confidence', 0):.2f}  "
            f"cx={p.get('x', 0):.0f}  cy={p.get('y', 0):.0f}  "
            f"w={p.get('width', 0):.0f}  h={p.get('height', 0):.0f}"
        )


def _build_boxes(predictions, img_w, img_h, pad=0.03):
    boxes = []
    for p in predictions:
        cx = float(p["x"])
        cy = float(p["y"])
        bw = float(p["width"])
        bh = float(p["height"])
        conf = float(p.get("confidence", 1.0))

        side = max(bw, bh) * (1.0 + 2.0 * pad)
        half = side / 2.0

        x1 = int(max(0, cx - half))
        y1 = int(max(0, cy - half))
        x2 = int(min(img_w, cx + half))
        y2 = int(min(img_h, cy + half))

        boxes.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cx": cx, "cy": cy, "conf": conf,
            "w": x2 - x1, "h": y2 - y1,
        })
    return boxes


def _cluster_rows(boxes, height_multiplier=0.70):
    if not boxes:
        return []

    sorted_y = sorted(boxes, key=lambda b: (b["y1"] + b["y2"]) / 2)
    rows = []
    current_row = [sorted_y[0]]

    for box in sorted_y[1:]:
        prev = current_row[-1]
        prev_cy = (prev["y1"] + prev["y2"]) / 2
        curr_cy = (box["y1"] + box["y2"]) / 2
        avg_h = ((prev["y2"] - prev["y1"]) + (box["y2"] - box["y1"])) / 2
        gap = abs(curr_cy - prev_cy)
        threshold = height_multiplier * avg_h

        print(
            f"   [cluster] prev_cy={prev_cy:.0f}  curr_cy={curr_cy:.0f}  "
            f"gap={gap:.0f}  threshold={threshold:.0f}  "
            f"{'→ NEW ROW' if gap > threshold else '→ same row'}"
        )

        if gap > threshold:
            rows.append(current_row)
            current_row = [box]
        else:
            current_row.append(box)

    rows.append(current_row)

    for i, row in enumerate(rows):
        rows[i] = sorted(row, key=lambda b: (b["x1"] + b["x2"]) / 2)

    rows.sort(key=lambda row: sum((b["y1"] + b["y2"]) / 2 for b in row) / len(row))
    return rows


def _select_meter_dials(rows, expected_gauges):
    """
    Selection order:
      D1 = bottom-left
      D2.. = top-row left -> right
    """
    if not rows:
        return []

    top_row = rows[0]
    bottom_row = rows[1] if len(rows) > 1 else []

    top_row = sorted(top_row, key=lambda b: (b["x1"] + b["x2"]) / 2)
    bottom_row = sorted(bottom_row, key=lambda b: (b["x1"] + b["x2"]) / 2)

    selected = []

    # Dial 1 = bottom-left
    if bottom_row:
        selected.append(bottom_row[0])

    # Remaining dials = top row left -> right
    needed_from_top = max(0, expected_gauges - len(selected))
    selected.extend(top_row[:needed_from_top])

    return selected[:expected_gauges]


def _save_debug_image(image, boxes, selected):
    vis = image.copy()
    for b in boxes:
        cv2.rectangle(vis, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (160, 160, 160), 1)
    for i, b in enumerate(selected):
        cv2.rectangle(vis, (b["x1"], b["y1"]), (b["x2"], b["y2"]), (0, 220, 0), 2)
        cv2.putText(
            vis, f"D{i+1}",
            (b["x1"] + 4, b["y1"] + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2
        )
    cv2.imwrite("debug_detections.png", vis)
    print("   Saved: debug_detections.png")


def d_main(image_path, expected_gauges=5, debug=False):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    img_h, img_w = image.shape[:2]

    print(f"\n🔍  Running Roboflow workflow on: {image_path}")
    result = _client.run_workflow(
        workspace_name=WORKSPACE_NAME,
        workflow_id=WORKFLOW_ID,
        images={"image": image_path},
        use_cache=True,
    )

    predictions = _find_predictions(result)
    if not predictions:
        print("⚠️  No predictions returned by workflow.")
        return []

    print(f"   Found {len(predictions)} raw prediction(s):")
    _print_predictions(predictions)

    boxes = _build_boxes(predictions, img_w, img_h, pad=0.03)

    print(f"\n   Clustering {len(boxes)} box(es) into rows ...")
    rows = _cluster_rows(boxes)
    if not rows:
        print("⚠️  No rows found after clustering.")
        return []

    print(f"\n   Clustered into {len(rows)} row(s):")
    for ri, row in enumerate(rows):
        avg_y = sum((b["y1"] + b["y2"]) / 2 for b in row) / len(row)
        xs = ", ".join(str(int(b["cx"])) for b in row)
        print(f"   Row {ri}: {len(row)} dial(s)  avg-y={avg_y:.0f}  cx=[{xs}]")

    selected = _select_meter_dials(rows, expected_gauges)

    print(f"\n   Selected {len(selected)} dial(s) in requested order (D1 bottom, then top left -> right):")
    for i, b in enumerate(selected):
        print(
            f"   Dial {i+1}: cx={b['cx']:.0f}  cy={b['cy']:.0f}  "
            f"box=[{b['x1']},{b['y1']} -> {b['x2']},{b['y2']}]  "
            f"conf={b['conf']:.2f}"
        )

    if debug:
        _save_debug_image(image, boxes, selected)

    cropped = []
    for i, b in enumerate(selected):
        crop = image[b["y1"]:b["y2"], b["x1"]:b["x2"]]
        if crop.size == 0:
            print(f"   ⚠️  Dial {i+1} crop is empty — skipping.")
            continue
        cropped.append(crop)
        if debug:
            out = f"debug_crop_{i+1}.png"
            cv2.imwrite(out, crop)
            print(f"   Saved: {out}")

    print(f"\n✅  Returning {len(cropped)} cropped dial(s).")
    return cropped
