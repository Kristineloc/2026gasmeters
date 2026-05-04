from dataclasses import dataclass
from typing import Optional, Tuple, List
import math


@dataclass
class MethodPrediction:
    source: str
    abs_angle: Optional[float]
    confidence: float
    digit: Optional[int] = None


def circular_diff_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def wrap360(angle: float) -> float:
    return angle % 360.0


def relative_angle(abs_ang: float, zero_angle: float = 0.0) -> float:
    return wrap360(abs_ang - zero_angle)


def to_unified_cw(rel_ang: float, counter_clockwise: bool) -> float:
    return wrap360(360.0 - rel_ang) if counter_clockwise else wrap360(rel_ang)


def digit_from_angle(rel_ang_cw: float) -> int:
    return int(rel_ang_cw // 36.0) % 10


def angle_to_digit(
    abs_angle: float,
    counter_clockwise: bool,
    zero_angle: float = 0.0,
) -> Tuple[int, float]:
    rel = relative_angle(abs_angle, zero_angle=zero_angle)
    rel_cw = to_unified_cw(rel, counter_clockwise=counter_clockwise)
    return digit_from_angle(rel_cw), rel_cw


def dial_value_from_angle(
    abs_angle: float,
    counter_clockwise: bool,
    zero_angle: float = 0.0,
) -> Tuple[float, int, float, float]:
    """
    Convert the measured needle angle into a continuous dial value in [0, 10),
    plus its lower digit and fractional part.

    Returns:
        value        continuous dial reading in [0, 10)
        lower_digit  floor(value)
        frac         fractional part of value
        rel_ang_cw   unified clockwise angle used for digit mapping
    """
    rel = relative_angle(abs_angle, zero_angle=zero_angle)
    rel_cw = to_unified_cw(rel, counter_clockwise=counter_clockwise)

    value = (rel_cw / 36.0) % 10.0
    lower_digit = int(math.floor(value + 1e-6)) % 10
    frac = value - math.floor(value)
    return value, lower_digit, frac, rel_cw


def normalize_method_predictions(
    predictions: List[MethodPrediction],
    counter_clockwise: bool,
    zero_angle: float = 0.0,
) -> List[MethodPrediction]:
    out: List[MethodPrediction] = []
    for pred in predictions:
        new_pred = MethodPrediction(
            source=pred.source,
            abs_angle=pred.abs_angle,
            confidence=float(max(0.0, min(1.0, pred.confidence))),
            digit=pred.digit,
        )
        if new_pred.digit is None and new_pred.abs_angle is not None:
            new_pred.digit, _ = angle_to_digit(
                new_pred.abs_angle,
                counter_clockwise,
                zero_angle,
            )
        out.append(new_pred)
    return out


def weighted_digit_vote(predictions, method_weights=None):
    if method_weights is None:
        method_weights = {"ray": 1.0, "contour": 0.9, "line": 0.8, "classifier": 1.3}

    usable = [p for p in predictions if p.digit is not None and p.confidence > 0.0]
    if not usable:
        return 0, {0: 0.0}, []

    score_by_digit = {d: 0.0 for d in range(10)}
    for pred in usable:
        score_by_digit[pred.digit] += method_weights.get(pred.source, 1.0) * pred.confidence

    best_digit = max(score_by_digit, key=score_by_digit.get)
    return best_digit, score_by_digit, usable


def ensemble_confidence(score_by_digit, chosen_digit):
    chosen = score_by_digit.get(chosen_digit, 0.0)
    others = [v for k, v in score_by_digit.items() if k != chosen_digit]
    second = max(others) if others else 0.0
    total = sum(score_by_digit.values()) + 1e-6
    margin = max(0.0, chosen - second) / total
    concentration = chosen / total
    return float(max(0.0, min(1.0, 0.55 * concentration + 0.45 * margin)))


def choose_digit_for_dial(predictions, counter_clockwise, zero_angle=0.0, method_weights=None):
    preds = normalize_method_predictions(predictions, counter_clockwise, zero_angle)
    digit, score_by_digit, usable = weighted_digit_vote(preds, method_weights=method_weights)
    conf = ensemble_confidence(score_by_digit, digit)
    return digit, conf, score_by_digit, usable


def on_digit(value: float, tol_digit: float = 0.08) -> bool:
    """
    Check whether the dial is effectively sitting on a whole number.
    tol_digit is in digit units, not degrees.
    0.08 digit is about 2.88 degrees.
    """
    frac = value % 1.0
    return frac <= tol_digit or frac >= (1.0 - tol_digit)


def passed_zero(right_value: float, zero_pass_threshold: float = 1.0) -> bool:
    """
    In the unified continuous-reading frame, a lower-order dial has passed zero
    when it has already moved into the interval [0, 1).

    Examples:
      0.2  -> passed zero
      9.8  -> has not passed zero yet
    """
    v = right_value % 10.0
    return v < zero_pass_threshold


def apply_rule4(digits_right_to_left, values_right_to_left, on_digit_tol: float = 0.08):
    """
    Apply the common gas meter rule:
    - If a dial is between two numbers, record the lower number.
    - If a dial is exactly on a number, look at the dial to the right.
      If the right dial has not yet passed zero, subtract 1.

    Inputs are in right-to-left order:
      index 0 = rightmost / lowest-order dial
      index 1 = next dial to the left
      ...
    """
    corrected = digits_right_to_left[:]

    for i in range(1, len(corrected)):
        current_digit = corrected[i]
        current_value = values_right_to_left[i]
        right_value = values_right_to_left[i - 1]

        if not on_digit(current_value, tol_digit=on_digit_tol):
            continue

        if not passed_zero(right_value):
            corrected[i] = (current_digit - 1) % 10

    return corrected
