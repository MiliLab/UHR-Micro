from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

Point = tuple[float, float]
Box = list[float]


def is_finite_box(box: Sequence[float]) -> bool:
    return len(box) in (4, 8) and all(math.isfinite(float(v)) for v in box)


def box_to_polygon(box: Sequence[float]) -> list[Point]:
    if len(box) == 4:
        x1, y1, x2, y2 = map(float, box)
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    if len(box) == 8:
        return [(float(box[i]), float(box[i + 1])) for i in range(0, 8, 2)]
    raise ValueError(f"box must have 4 or 8 values, got {len(box)}")


def signed_area(points: Sequence[Point]) -> float:
    return 0.5 * sum(
        points[i][0] * points[(i + 1) % len(points)][1]
        - points[(i + 1) % len(points)][0] * points[i][1]
        for i in range(len(points))
    )


def polygon_area(points: Sequence[Point]) -> float:
    return abs(signed_area(points))


def _cross(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])


def is_convex_quad(points: Sequence[Point], eps: float = 1e-9) -> bool:
    if len(points) != 4 or polygon_area(points) <= eps:
        return False
    crosses = [_cross(points[i], points[(i + 1) % 4], points[(i + 2) % 4]) for i in range(4)]
    return all(v > eps for v in crosses) or all(v < -eps for v in crosses)


def normalize_ccw(points: Sequence[Point]) -> list[Point]:
    out = list(points)
    return out if signed_area(out) > 0 else list(reversed(out))


def _inside(p: Point, a: Point, b: Point, eps: float = 1e-9) -> bool:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= -eps


def _intersection(s: Point, e: Point, a: Point, b: Point) -> Point:
    dx1, dy1 = e[0] - s[0], e[1] - s[1]
    dx2, dy2 = b[0] - a[0], b[1] - a[1]
    den = dx1 * dy2 - dy1 * dx2
    if abs(den) < 1e-12:
        return e
    t = ((a[0] - s[0]) * dy2 - (a[1] - s[1]) * dx2) / den
    return (s[0] + t * dx1, s[1] + t * dy1)


def convex_intersection(subject: Sequence[Point], clip: Sequence[Point]) -> list[Point]:
    output = normalize_ccw(subject)
    clip_ccw = normalize_ccw(clip)
    for i, a in enumerate(clip_ccw):
        b = clip_ccw[(i + 1) % len(clip_ccw)]
        inp, output = output, []
        if not inp:
            break
        s = inp[-1]
        for e in inp:
            e_in, s_in = _inside(e, a, b), _inside(s, a, b)
            if e_in:
                if not s_in:
                    output.append(_intersection(s, e, a, b))
                output.append(e)
            elif s_in:
                output.append(_intersection(s, e, a, b))
            s = e
    return output


def polygon_iou(left: Sequence[float], right: Sequence[float]) -> float:
    a, b = box_to_polygon(left), box_to_polygon(right)
    if len(left) == 8 and not is_convex_quad(a):
        return 0.0
    if len(right) == 8 and not is_convex_quad(b):
        return 0.0
    area_a, area_b = polygon_area(a), polygon_area(b)
    inter = polygon_area(convex_intersection(a, b))
    union = area_a + area_b - inter
    return max(0.0, min(1.0, inter / union)) if union > 0 else 0.0


def centroid(box: Sequence[float]) -> Point:
    pts = box_to_polygon(box)
    area2 = 2.0 * signed_area(pts)
    if abs(area2) < 1e-12:
        return (sum(x for x, _ in pts) / len(pts), sum(y for _, y in pts) / len(pts))
    cx = sum((pts[i][0] + pts[(i + 1) % len(pts)][0]) * (
        pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1]
    ) for i in range(len(pts))) / (3.0 * area2)
    cy = sum((pts[i][1] + pts[(i + 1) % len(pts)][1]) * (
        pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1]
    ) for i in range(len(pts))) / (3.0 * area2)
    return (cx, cy)


def bbox_diagonal(box: Sequence[float]) -> float:
    pts = box_to_polygon(box)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return max(math.hypot(max(xs) - min(xs), max(ys) - min(ys)), 1.0)


def side_lengths(box: Sequence[float]) -> tuple[float, float]:
    pts = box_to_polygon(box)
    lengths = [math.hypot(pts[(i + 1) % 4][0] - pts[i][0], pts[(i + 1) % 4][1] - pts[i][1]) for i in range(4)]
    return ((lengths[0] + lengths[2]) / 2.0, (lengths[1] + lengths[3]) / 2.0)


def scale_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    lw, lh = side_lengths(left)
    rw, rh = side_lengths(right)
    def score(a: float, b: float, c: float, d: float) -> float:
        sw = min(a, c) / max(a, c) if max(a, c) > 0 else 0.0
        sh = min(b, d) / max(b, d) if max(b, d) > 0 else 0.0
        return math.sqrt(sw * sh)
    return max(score(lw, lh, rw, rh), score(lw, lh, rh, rw))


def validate_box(box: Sequence[float], width: float, height: float, *, range_check: bool = True) -> tuple[bool, str]:
    if not is_finite_box(box):
        return False, "nonfinite_failure"
    if range_check:
        for i, value in enumerate(box):
            limit = width if i % 2 == 0 else height
            if float(value) < 0.0 or float(value) > float(limit):
                return False, "range_failure"
    if len(box) == 4:
        if not (float(box[2]) > float(box[0]) and float(box[3]) > float(box[1])):
            return False, "order_failure"
    elif not is_convex_quad(box_to_polygon(box)):
        return False, "order_failure"
    return True, "ok"


def _size(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, dict):
        return None
    try:
        width, height = float(value["width_px"]), float(value["height_px"])
    except (KeyError, TypeError, ValueError):
        return None
    return (width, height) if width > 0 and height > 0 else None


def resolve_coordinate_protocol(requested: str, diagnostics: dict[str, Any]) -> str | None:
    if requested == "explicit_prompt_frozen":
        value = str(diagnostics.get("explicit_coordinate_protocol") or "")
        return value if value in {"base1000", "base100", "base1", "absolute_source", "absolute_model_input"} else None
    return requested


def transform_boxes_to_source(
    boxes: Iterable[Sequence[float]], requested_protocol: str, source_size: dict[str, Any],
    model_input_size: dict[str, Any] | None, diagnostics: dict[str, Any],
) -> tuple[list[Box], str, dict[str, Any]]:
    source = _size(source_size)
    if source is None:
        return [], "canvas_failure", {"reason": "missing_source_size"}
    width, height = source
    protocol = resolve_coordinate_protocol(requested_protocol, diagnostics)
    if protocol is None or protocol == "pending_map_v4_contract":
        return [], "canvas_failure", {"reason": "unresolved_coordinate_protocol"}
    transformed: list[Box] = []
    for raw in boxes:
        values = [float(v) for v in raw]
        if not is_finite_box(values):
            return [], "nonfinite_failure", {"reason": "nonfinite_or_bad_arity"}
        if protocol in {"base1000", "base100", "base1"}:
            base = {"base1000": 1000.0, "base100": 100.0, "base1": 1.0}[protocol]
            if any(v < 0 or v > base for v in values):
                return [], "range_failure", {"reason": "declared_base_range", "base": base}
            out = [v * (width if i % 2 == 0 else height) / base for i, v in enumerate(values)]
        elif protocol == "absolute_source":
            out = values
        elif protocol == "absolute_model_input":
            model_size = _size(model_input_size)
            transform = diagnostics.get("source_from_model_input")
            if not model_size or not isinstance(transform, dict):
                return [], "canvas_failure", {"reason": "absolute_model_input_requires_explicit_affine"}
            try:
                sx, sy = float(transform["scale_x"]), float(transform["scale_y"])
                ox, oy = float(transform["offset_x"]), float(transform["offset_y"])
            except (KeyError, TypeError, ValueError):
                return [], "canvas_failure", {"reason": "bad_source_from_model_input_affine"}
            mw, mh = model_size
            if any(v < 0 or v > (mw if i % 2 == 0 else mh) for i, v in enumerate(values)):
                return [], "range_failure", {"reason": "model_input_range"}
            out = [v * (sx if i % 2 == 0 else sy) + (ox if i % 2 == 0 else oy) for i, v in enumerate(values)]
        else:
            return [], "canvas_failure", {"reason": f"unsupported_protocol:{protocol}"}
        ok, status = validate_box(out, width, height)
        if not ok:
            return [], status, {"reason": "transformed_box_invalid", "protocol": protocol}
        transformed.append(out)
    return transformed, "ok", {"resolved_coordinate_protocol": protocol}
