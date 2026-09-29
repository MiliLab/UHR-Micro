#!/usr/bin/env python3
"""Create the answer-free public contract and a sealed reporting binding.

Only the public question is opened from each QA JSON.  The reporting binding is
written separately and is never consumed by inference-time code.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import secrets
from collections import Counter
from pathlib import Path
from typing import Any

from io_utils import read_jsonl, sha256_file, sha256_text, write_json, write_jsonl


LIST_RE = re.compile(r"\[([^\[\]]+)\]")
VALIDATION_MANIFEST_SHA256 = (
    "2549d15d5b405b799ed395b3382dd895e703637e5af2c79589e0a0eda29427f3"
)
OPAQUE_ID_RE = re.compile(r"^[0-9a-f]{64}$")
OPTION_RE = re.compile(r"(?:^|\n)\s*A\s*[\.:]", re.IGNORECASE)
NUMERIC_COMPARISON_RE = re.compile(
    r"\babsolute\s+difference\b|\bdifference\s+between\b|\bcompare\b.*\bnumber\b",
    re.IGNORECASE | re.DOTALL,
)
FINE_DETAIL_RE = re.compile(
    r"\bmodel\s+type\b|\bfine[- ]grained\b|\bsubtype\b|\bvariant\b",
    re.IGNORECASE,
)
GLOBAL_CONFIGURATION_RE = re.compile(
    r"\bnearest\b|\bfarthest\b|\bclosest\b|\bspatial\s+distribution\b|"
    r"\bdistribution\s+of\b|\boverall\s+distribution\b",
    re.IGNORECASE,
)
PAIRWISE_CONFIGURATION_RE = re.compile(
    r"\brelative\s+to\b.*\bdirection\b|\bdirection\s+of\b.*\brelative\s+to\b",
    re.IGNORECASE | re.DOTALL,
)
OBJECT_PART_RE = re.compile(
    r"\b(?:left|right)?\s*(?:wing|engine)\b|\bfuselage\b|\btail\s+fins?\b|"
    r"\bcargo\s+body\b|\bbridge\s+tower\b|\bthreshold\s+marking\b|"
    r"\bcomponent\b|\bwindshield\b|\bblade\b|\bbow\b|\bcab\b|"
    r"\bpitcher[’']?s\s+mound\b",
    re.IGNORECASE,
)


OPERATOR_ACTIVATION_ORDER = (
    "SUPPORT_CONTEXT",
    "TARGET_DETAIL",
    "GLOBAL_LAYOUT",
    "SEMANTIC_SCOPE",
    "NUMERIC_DIFFERENCE",
    "GEOMETRY_FUSION",
    "EVIDENCE_CLOSURE",
    "MASK_ADAPTER",
)
GEOMETRY_SCHEMAS = frozenset({"SINGLE_GEOMETRY", "GEOMETRY_SET"})
PUBLIC_SUPPORT_SOURCES = frozenset(
    {
        "PUBLIC_EXPLICIT_SUPPORT",
        "PUBLIC_TARGET_SUPPORT",
        "PUBLIC_SEMANTIC_SUPPORT",
        "MODEL_PROPOSED_SUPPORT",
    }
)
PUBLIC_TARGET_REFERENCE_RE = re.compile(
    r"\bobject\s+indicated\s+by\s+(?:the\s+)?(?:"
    r"(?:(?:axis[- ]aligned|oriented)\s+)?bounding\s+box"
    r"(?:\s+with\s+vertex\s+coordinates)?|"
    r"center\s+point(?:\s+with\s+vertex\s+coordinates)?"
    r")\s*\[",
    re.IGNORECASE,
)
BOUNDED_ONE_LIST_CLAUSE_RE = re.compile(
    r"\s+(?:in|near|in/near)\s+(?:the\s+)?region\s+of\s*\[[^\[\]]+\]",
    re.IGNORECASE,
)
BOUNDED_TWO_CORNER_CLAUSE_RE = re.compile(
    r"\s+within\s+the\s+rectangular\s+area\s+defined\s+by\s+pixels\s*"
    r"\[[^\[\]]+\]\s+and\s+\[[^\[\]]+\]",
    re.IGNORECASE,
)
# This compatibility predicate deliberately excludes an indicated target box:
# target geometry identifies the object itself, whereas these clauses delimit
# a search area inside which another object or set must be observed.
EXPLICIT_REGION_RE = re.compile(
    r"\b(?:in|near|in/near)\s+(?:the\s+)?region\s+of\s*\[|"
    r"\bwithin\s+the\s+rectangular\s+area\s+defined\s+by\s+pixels\s*\[",
    re.IGNORECASE,
)
ALL_INSTANCES_RE = re.compile(
    r"\b(?:count|detect|locate)\s+all\s+instances\s+of\b", re.IGNORECASE
)
FULL_SCENE_RE = re.compile(
    r"\b(?:count|detect|locate)\s+all\s+instances\s+of\s+(.+?)\s+in\s+this\s+remote\s+sensing\s+image\b",
    re.IGNORECASE | re.DOTALL,
)
TWO_CORNER_REGION_RE = re.compile(
    r"rectangular\s+area\s+defined\s+by\s+pixels\s*"
    r"\[([^\[\]]+)\]\s+and\s+\[([^\[\]]+)\]",
    re.IGNORECASE,
)
AXIS_THRESHOLD_RE = re.compile(
    r"center\s+([xy])-coordinate\s+in\s+the\s+original\s+image\s+is\s+at\s+or\s+"
    r"(left|right|above|below)(?:\s+of)?\s+(-?(?:\d+(?:\.\d*)?|\.\d+))\s+pixels?",
    re.IGNORECASE,
)

FINAL_INSTRUCTION_PATTERNS = [
    r"\s+(?:and\s+)?return\s+their\s+locations\s+as\s+(?:axis-aligned|oriented)\s+bounding\s+boxes,\s*where\s+each\s+box\s+is\s+represented\s+by\s+.*?\[[^\]]+\]\.?\s*$",
    r"\s+return\s+its\s+location\s+as\s+(?:a|an)\s+(?:axis-aligned\s+|oriented\s+)?bounding\s+box\s+represented\s+by\s+\[[^\]]+\]\.?\s*$",
    r"\s+please\s+output\s+only\s+the\s+letter\s+of\s+the\s+correct\s+option\s+\(A,\s*B,\s*C,\s*or\s*D\)\.?\s*$",
    r"\s+(?:and\s+)?return\s+only\s+one\s+Arabic\s+numeral\.?\s*$",
    r"\s+please\s+answer\s+with\s+one\s+Arabic\s+numeral\.?\s*$",
    r"\s+return\s+the\s+number\s+only\.?\s*$",
]
TERMINAL_OUTPUT_RESIDUAL_RE = re.compile(
    r"\breturn\s+(?:its|their)\s+locations?\s+as\s+.*?bounding\s+boxes?\b|"
    r"\breturn\s+only\s+one\s+Arabic\s+numeral\b|"
    r"\bplease\s+answer\s+with\s+one\s+Arabic\s+numeral\b|"
    r"\breturn\s+the\s+number\s+only\b|"
    r"\bplease\s+output\s+only\s+the\s+letter\s+of\s+the\s+correct\s+option\b",
    re.IGNORECASE | re.DOTALL,
)


def infer_answer_schema(question: str) -> str:
    lower = question.lower()
    if OPTION_RE.search(question):
        return "OPTION"
    if (
        "their locations" in lower
        or "bounding boxes" in lower
        or "each box is represented" in lower
    ):
        return "GEOMETRY_SET"
    if "arabic numeral" in lower or re.search(r"\bcount\b", lower):
        return "SCALAR"
    if "bounding box" in lower or "location as" in lower:
        return "SINGLE_GEOMETRY"
    raise ValueError(f"cannot infer public answer schema: {question}")


def infer_geometry_encoding(question: str, schema: str) -> str:
    if schema not in {"SINGLE_GEOMETRY", "GEOMETRY_SET"}:
        return "NONE"
    if re.search(
        r"oriented\s+bounding\s+box|4\s+polygon\s+points|"
        r"\[\s*x1\s*,\s*y1\s*,\s*x2\s*,\s*y2\s*,\s*x3\s*,\s*y3\s*,\s*x4\s*,\s*y4\s*\]",
        question,
        re.IGNORECASE,
    ):
        return "QUAD8"
    return "AABB4"


def numeric_lists(question: str) -> list[list[float]]:
    output: list[list[float]] = []
    for match in LIST_RE.finditer(question):
        parts = [part.strip() for part in match.group(1).split(",") if part.strip()]
        if len(parts) < 2 or len(parts) % 2:
            continue
        try:
            values = [float(part) for part in parts]
        except ValueError:
            continue
        if all(math.isfinite(value) for value in values):
            output.append(values)
    return output


def validate_absolute(values: list[float], width: int, height: int) -> None:
    for index, value in enumerate(values):
        upper = float(width if index % 2 == 0 else height)
        if not 0.0 <= value <= upper:
            axis = "x" if index % 2 == 0 else "y"
            raise ValueError(
                f"public {axis} coordinate {value} outside source boundary [0,{upper}]"
            )


def source1000(values: list[float], width: int, height: int) -> list[float]:
    validate_absolute(values, width, height)
    return [
        round(value / float(width if index % 2 == 0 else height) * 1000.0, 1)
        for index, value in enumerate(values)
    ]


def repair_public_region(
    values: list[float], width: int, height: int
) -> tuple[list[float], list[dict[str, float | int | str]]]:
    """Intersect a public region with the source frame and record every repair.

    This boundary repair is permitted only for boxes or polygons that overlap
    the image.  It is never used for model outputs and is never silent.
    """

    if len(values) == 2:
        validate_absolute(values, width, height)
        return values, []
    repaired: list[float] = []
    records: list[dict[str, float | int | str]] = []
    for index, value in enumerate(values):
        upper = float(width if index % 2 == 0 else height)
        clipped = max(0.0, min(upper, value))
        repaired.append(clipped)
        if clipped != value:
            records.append(
                {
                    "coordinate_index": index,
                    "axis": "x" if index % 2 == 0 else "y",
                    "original": value,
                    "repaired": clipped,
                    "boundary": upper,
                }
            )
    candidate = repaired if len(repaired) == 4 else [
        min(repaired[0::2]),
        min(repaired[1::2]),
        max(repaired[0::2]),
        max(repaired[1::2]),
    ]
    if not (candidate[2] > candidate[0] and candidate[3] > candidate[1]):
        raise ValueError(f"public region does not overlap the source frame: {values}")
    validate_absolute(repaired, width, height)
    return repaired, records


def render_numbers(values: list[float]) -> str:
    def render(value: float) -> str:
        return str(int(round(value))) if abs(value - round(value)) < 1e-8 else f"{value:.4f}".rstrip("0").rstrip(".")

    return "[" + ",".join(render(value) for value in values) + "]"


def convert_question_coordinates(
    question: str, width: int, height: int
) -> tuple[
    str,
    list[list[float]],
    list[list[float]],
    list[dict[str, Any]],
]:
    absolute: list[list[float]] = []
    normalized: list[list[float]] = []
    repairs: list[dict[str, Any]] = []

    def replace(match: re.Match[str]) -> str:
        parts = [part.strip() for part in match.group(1).split(",") if part.strip()]
        if len(parts) < 2 or len(parts) % 2:
            return match.group(0)
        try:
            values = [float(part) for part in parts]
        except ValueError:
            return match.group(0)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("non-finite public coordinate")
        repaired, coordinate_repairs = repair_public_region(values, width, height)
        mapped = source1000(repaired, width, height)
        absolute.append(repaired)
        normalized.append(mapped)
        if coordinate_repairs:
            repairs.append(
                {
                    "original_values": values,
                    "repaired_values": repaired,
                    "coordinate_repairs": coordinate_repairs,
                }
            )
        return render_numbers(mapped)

    return LIST_RE.sub(replace, question), absolute, normalized, repairs


def convert_axis_thresholds(
    question: str, width: int, height: int
) -> tuple[str, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []

    def replace(match: re.Match[str]) -> str:
        axis = match.group(1).lower()
        relation = match.group(2).lower()
        value = float(match.group(3))
        upper = float(width if axis == "x" else height)
        if not math.isfinite(value) or not 0.0 <= value <= upper:
            raise ValueError(
                f"public {axis}-axis threshold {value} outside source boundary [0,{upper}]"
            )
        mapped = round(value / upper * 1000.0, 1)
        records.append(
            {
                "axis": axis,
                "relation": relation,
                "value_abs": value,
                "value_source1000": mapped,
            }
        )
        relation_phrase = (
            f"{relation} of" if relation in {"left", "right"} else relation
        )
        return (
            f"center {axis}-coordinate in the source-image 0-1000 coordinate "
            f"system is at or {relation_phrase} {render_numbers([mapped])[1:-1]}"
        )

    return AXIS_THRESHOLD_RE.sub(replace, question), records


def ordered_box(values: list[float], width: int, height: int) -> list[int]:
    if len(values) >= 6 and len(values) % 2 == 0:
        xs, ys = values[0::2], values[1::2]
        raw = [min(xs), min(ys), max(xs), max(ys)]
    elif len(values) == 4:
        raw = values
    else:
        raise ValueError("public support requires four box coordinates or an even polygon vertex list")
    validate_absolute(raw, width, height)
    x0, x1 = sorted((raw[0], raw[2]))
    y0, y1 = sorted((raw[1], raw[3]))
    result = [int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))]
    if not (0 <= result[0] < result[2] <= width and 0 <= result[1] < result[3] <= height):
        raise ValueError(f"degenerate public support box: {result}")
    return result


def point_window(values: list[float], width: int, height: int) -> list[int]:
    validate_absolute(values, width, height)
    side = max(1, min(1024, width, height))
    cx, cy = values
    x0 = min(max(int(round(cx - side / 2)), 0), max(0, width - side))
    y0 = min(max(int(round(cy - side / 2)), 0), max(0, height - side))
    return [x0, y0, x0 + side, y0 + side]


def explicit_support_boxes(
    question: str, values: list[list[float]], width: int, height: int
) -> list[list[int]]:
    # Two-corner public rectangles are one support, not two point-centred ROIs.
    consumed_point_pairs: set[tuple[float, ...]] = set()
    output: list[list[int]] = []
    for match in TWO_CORNER_REGION_RE.finditer(question):
        corners: list[list[float]] = []
        for raw in match.groups():
            parts = [part.strip() for part in raw.split(",") if part.strip()]
            try:
                corner = [float(part) for part in parts]
            except ValueError:
                corners = []
                break
            if len(corner) != 2:
                corners = []
                break
            corners.append(corner)
        if len(corners) == 2:
            output.append(ordered_box(corners[0] + corners[1], width, height))
            consumed_point_pairs.update(tuple(corner) for corner in corners)
    for item in values:
        if len(item) == 2:
            if tuple(item) not in consumed_point_pairs:
                output.append(point_window(item, width, height))
        else:
            output.append(ordered_box(item, width, height))
    unique: list[list[int]] = []
    for box in output:
        if box not in unique:
            unique.append(box)
    return unique


def strip_answer_instruction(question: str) -> str:
    output = question.strip()
    for pattern in FINAL_INSTRUCTION_PATTERNS:
        output = re.sub(pattern, "", output, flags=re.IGNORECASE)
    return re.sub(r"\s+(?:and|to)\s*$", "", output, flags=re.IGNORECASE).strip()


def pairwise_support_required(question: str, schema: str) -> bool:
    return bool(NUMERIC_COMPARISON_RE.search(question)) or (
        schema == "OPTION" and bool(PAIRWISE_CONFIGURATION_RE.search(question))
    )


def public_image_alias(source: Path, image_sha256: str, alias_root: Path) -> Path:
    """Expose image bytes through a content-addressed, task-opaque hard link."""

    alias_root.mkdir(parents=True, exist_ok=True)
    suffix = source.suffix.lower()
    if suffix not in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}:
        suffix = ".img"
    target = alias_root / f"{image_sha256}{suffix}"
    if target.exists():
        if sha256_file(target) != image_sha256:
            raise ValueError(f"content-addressed image alias is corrupt: {target}")
        return target.absolute()
    try:
        os.link(source, target)
    except FileExistsError:
        pass
    if sha256_file(target) != image_sha256:
        raise ValueError(f"content-addressed image alias hash mismatch: {target}")
    return target.absolute()


def complete_scene_enumeration(question: str) -> bool:
    match = FULL_SCENE_RE.search(question)
    if not match:
        return False
    target = match.group(1).strip().lower()
    return " in " not in target and " within " not in target


def semantic_region_enumeration(question: str) -> bool:
    """Identify a public natural-language support domain without coordinates.

    The frozen benchmark uses the form ``all instances of <target> in
    <semantic region> in this remote sensing image``.  The same outer grammar
    without the inner ``in`` denotes a true full-scene enumeration.  This
    distinction is made entirely from the public question and is independent
    of task labels or historical result versions.
    """

    match = FULL_SCENE_RE.search(question)
    if not match:
        return False
    scoped_target = match.group(1).strip().lower()
    return " in " in scoped_target or " within " in scoped_target


def observation_geometry_encoding(answer_schema: str) -> str:
    """Return the model-facing evidence encoding, not final serialization.

    A local observer may describe visible geometry as either a clear
    axis-aligned box or a clear quadrilateral. Registration converts that
    observation into the source frame; the terminal boundary later enforces
    the encoding declared by the public question.
    """

    if answer_schema in GEOMETRY_SCHEMAS:
        return "AABB4_OR_QUAD8"
    return "NONE"


def compile_operator_activations(contract: dict[str, Any]) -> list[str]:
    """Compile the generic operator bank from public evidence requirements.

    Only the listed public contract fields are read. Extra metadata --
    including a task, path, historical-version, or metric field accidentally
    attached by a caller -- therefore cannot affect the execution plan.
    """

    required = {
        "answer_schema",
        "terminal_geometry_encoding",
        "support_topology",
        "support_extent",
        "coverage_requirement",
        "composition",
        "resolution_requirement",
        "support_source",
        "explicit_support_count",
    }
    missing = sorted(required - set(contract))
    if missing:
        raise ValueError(f"public evidence contract misses operator fields: {missing}")

    schema = str(contract["answer_schema"])
    terminal_encoding = str(contract["terminal_geometry_encoding"])
    extent = str(contract["support_extent"])
    coverage = str(contract["coverage_requirement"])
    composition = str(contract["composition"])
    resolution = str(contract["resolution_requirement"])
    support_source = str(contract["support_source"])
    explicit_count = int(contract["explicit_support_count"])

    if schema not in {"SCALAR", "OPTION", *GEOMETRY_SCHEMAS}:
        raise ValueError(f"unsupported public answer schema: {schema}")
    if schema in GEOMETRY_SCHEMAS and terminal_encoding not in {"AABB4", "QUAD8"}:
        raise ValueError(
            f"geometry schema requires a public AABB4 or QUAD8 terminal encoding: {terminal_encoding}"
        )
    if schema not in GEOMETRY_SCHEMAS and terminal_encoding != "NONE":
        raise ValueError(
            f"non-geometry schema requires terminal geometry encoding NONE: {terminal_encoding}"
        )
    if support_source not in PUBLIC_SUPPORT_SOURCES:
        raise ValueError(f"unsupported public support source: {support_source}")
    if explicit_count < 0:
        raise ValueError("explicit public support count cannot be negative")

    flags: set[str] = set()

    # Preserve the whole question-relevant support whenever the requested
    # evidence is not a complete-scene enumeration. This applies equally to
    # explicit and question-proposed support, never to named benchmark tasks.
    if extent != "FULL_SCENE" and coverage in {
        "LOCAL_SUPPORT",
        "COMPLETE_SUPPORT",
        "SEMANTIC_COMPLETE_SUPPORT",
    }:
        flags.add("SUPPORT_CONTEXT")

    # Fine subtype and object-part observations need detail magnification,
    # irrespective of how the common acquisition stage obtained the support.
    if resolution in {"FINE_SUBTYPE", "OBJECT_PART"}:
        flags.add("TARGET_DETAIL")

    # Every spatial-configuration question needs registered global layout;
    # pairwise layout is not treated as a benchmark-specific exception.
    if composition == "SPATIAL_CONFIGURATION" and resolution == "SCENE_LAYOUT":
        flags.add("GLOBAL_LAYOUT")

    # A region described only in natural language cannot be proven complete
    # by a finite set of point-centred crops.  Preserve the byte-unmodified
    # complete image for semantic localization; proposed local views remain
    # detail evidence and can never close this support on their own.
    if support_source == "PUBLIC_SEMANTIC_SUPPORT" and extent == "SEMANTIC_REGION":
        flags.add("SEMANTIC_SCOPE")

    if composition == "NUMERIC_DIFFERENCE":
        flags.add("NUMERIC_DIFFERENCE")

    if schema in GEOMETRY_SCHEMAS:
        flags.add("GEOMETRY_FUSION")

    # Evidence closure is a universal graph stage.  Its deterministic gate may
    # close a row or leave it open, but it is never an implicit task route.
    flags.add("EVIDENCE_CLOSURE")

    # Any public single-object AABB4 is eligible to seed a later mask boundary.
    # The public wording does not distinguish an AABB4 grounding answer from an
    # AABB4 mask prompt, so *all* identical public interfaces receive the same
    # eligibility. This flag never changes inference; the private reporting
    # evaluator decides whether to consume the geometry after inference.
    if schema == "SINGLE_GEOMETRY" and terminal_encoding == "AABB4":
        flags.add("MASK_ADAPTER")

    # Ordering is a frozen execution convention and makes hashes auditable.
    return [name for name in OPERATOR_ACTIVATION_ORDER if name in flags]


def compile_evidence_contract(
    question: str, schema: str, boxes: list[list[int]], geometry_encoding: str
) -> dict[str, Any]:
    comparison = bool(NUMERIC_COMPARISON_RE.search(question))
    fine = schema == "OPTION" and bool(FINE_DETAIL_RE.search(question))
    global_configuration = schema == "OPTION" and bool(
        GLOBAL_CONFIGURATION_RE.search(question)
    )
    pairwise_configuration = schema == "OPTION" and bool(
        PAIRWISE_CONFIGURATION_RE.search(question)
    )
    # A public coordinate is not sufficient to decide what it means.  The
    # public wording must distinguish geometry identifying the target itself
    # (classification/fine-grained recognition) from geometry bounding a
    # search area (region enumeration/component localization).  This is a
    # semantic interface decision, never a benchmark-task route.
    target_reference = bool(boxes) and bool(
        PUBLIC_TARGET_REFERENCE_RE.search(question)
    )
    bounded_search = bool(boxes) and bool(
        BOUNDED_ONE_LIST_CLAUSE_RE.search(question)
        or BOUNDED_TWO_CORNER_CLAUSE_RE.search(question)
    )
    if target_reference and bounded_search:
        raise ValueError("public support cannot be both target geometry and search region")
    explicit = bool(boxes) and bool(EXPLICIT_REGION_RE.search(question))
    complete_scene = (
        schema in {"SCALAR", "GEOMETRY_SET"}
        and not explicit
        and complete_scene_enumeration(question)
    )
    semantic_region = (
        schema in {"SCALAR", "GEOMETRY_SET"}
        and not explicit
        and semantic_region_enumeration(question)
    )
    enumeration = schema in {"SCALAR", "GEOMETRY_SET"} and bool(
        ALL_INSTANCES_RE.search(question)
    )

    if comparison:
        topology, extent, coverage, composition = (
            "PAIRWISE_SUPPORT",
            "MULTIPLE_SUPPORTS",
            "COMPLETE_SUPPORT",
            "NUMERIC_DIFFERENCE",
        )
    elif global_configuration:
        topology, extent, coverage, composition = (
            "SCENE_FIELD",
            "FULL_SCENE",
            "COMPLETE_SCENE",
            "SPATIAL_CONFIGURATION",
        )
    elif pairwise_configuration:
        topology, extent, coverage, composition = (
            "PAIRWISE_SUPPORT",
            "MULTIPLE_SUPPORTS",
            "COMPLETE_SUPPORT",
            "SPATIAL_CONFIGURATION",
        )
    elif complete_scene:
        topology, extent, coverage, composition = (
            "SET",
            "FULL_SCENE",
            "COMPLETE_SCENE",
            "AGGREGATION",
        )
    elif semantic_region:
        topology, extent, coverage, composition = (
            "SET",
            "SEMANTIC_REGION",
            "SEMANTIC_COMPLETE_SUPPORT",
            "AGGREGATION",
        )
    elif bounded_search and schema == "SINGLE_GEOMETRY":
        topology, extent, coverage, composition = (
            "SINGLE_SUPPORT",
            "BOUNDED_SUPPORT",
            "LOCAL_SUPPORT",
            "IDENTITY",
        )
    elif schema in {"SCALAR", "GEOMETRY_SET"} or enumeration:
        topology, extent, coverage, composition = (
            "SET",
            "BOUNDED_SUPPORT",
            "COMPLETE_SUPPORT",
            "AGGREGATION",
        )
    else:
        topology = "PAIRWISE_SUPPORT" if pairwise_configuration else "SINGLE_SUPPORT"
        extent = "SINGLE_TARGET" if boxes or schema == "SINGLE_GEOMETRY" else "BOUNDED_SUPPORT"
        coverage, composition = "LOCAL_SUPPORT", "IDENTITY"

    if fine:
        resolution = "FINE_SUBTYPE"
    elif OBJECT_PART_RE.search(question):
        resolution = "OBJECT_PART"
    elif global_configuration or pairwise_configuration:
        resolution = "SCENE_LAYOUT"
    else:
        resolution = "OBJECT"

    if bounded_search:
        support_source = "PUBLIC_EXPLICIT_SUPPORT"
    elif target_reference:
        support_source = "PUBLIC_TARGET_SUPPORT"
    elif semantic_region:
        support_source = "PUBLIC_SEMANTIC_SUPPORT"
    elif boxes and schema in {"SCALAR", "GEOMETRY_SET"}:
        support_source = "PUBLIC_EXPLICIT_SUPPORT"
    elif boxes and schema in {"OPTION", "SINGLE_GEOMETRY"}:
        support_source = "PUBLIC_TARGET_SUPPORT"
    else:
        support_source = "MODEL_PROPOSED_SUPPORT"

    if support_source in {"PUBLIC_EXPLICIT_SUPPORT", "PUBLIC_TARGET_SUPPORT"}:
        max_points = 0
    elif extent == "FULL_SCENE" or schema == "GEOMETRY_SET":
        max_points = 8
    elif extent == "MULTIPLE_SUPPORTS":
        max_points = 4
    else:
        max_points = 2

    # One fixed local-view ceiling is used for every public interface. Actual
    # view count follows support geometry/cardinality, never a task or score.
    max_views = 8

    # Every public interface executes the same MAP observation chain. The
    # generic operator bank is compiled from this public evidence contract;
    # there is no named-task route or historical-version lookup.
    observations = ["NATIVE_LOCAL"]
    terminal_encoding = geometry_encoding
    model_observation_encoding = observation_geometry_encoding(schema)

    contract = {
        "answer_schema": schema,
        # Compatibility alias for existing output code. New code must use the
        # explicit observation and terminal fields below.
        "geometry_encoding": geometry_encoding,
        "observation_geometry_encoding": model_observation_encoding,
        "terminal_geometry_encoding": terminal_encoding,
        "support_topology": topology,
        "support_extent": extent,
        "coverage_requirement": coverage,
        "composition": composition,
        "resolution_requirement": resolution,
        "support_source": support_source,
        "explicit_support_count": len(boxes),
        "max_support_points": max_points,
        "max_native_views": max_views,
        "observation_operators": observations,
    }
    contract["operator_activations"] = compile_operator_activations(contract)
    return contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--public-out", type=Path, required=True)
    parser.add_argument("--private-binding-out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    parser.add_argument("--identity-map", type=Path, required=True)
    parser.add_argument(
        "--expected-manifest-sha256", default=VALIDATION_MANIFEST_SHA256
    )
    args = parser.parse_args()

    import ijson

    manifest_sha256 = sha256_file(args.manifest)
    manifest_rows = read_jsonl(args.manifest)
    if args.identity_map.is_file():
        identity_rows = read_jsonl(args.identity_map)
        identities = {str(row["qa_relpath"]): str(row["sample_id"]) for row in identity_rows}
    else:
        identities = {
            str(row["qa_relpath"]): secrets.token_hex(32) for row in manifest_rows
        }
        write_jsonl(
            args.identity_map,
            [
                {"qa_relpath": str(row["qa_relpath"]), "sample_id": identities[str(row["qa_relpath"])]}
                for row in manifest_rows
            ],
        )
    if set(identities) != {str(row["qa_relpath"]) for row in manifest_rows}:
        raise ValueError("private opaque identity map does not match manifest")
    identity_values = list(identities.values())
    if (
        len(identity_values) != len(set(identity_values))
        or not all(OPAQUE_ID_RE.fullmatch(value) for value in identity_values)
        or any(
            value == sha256_text(path)
            for path, value in identities.items()
        )
    ):
        raise PermissionError("sample identities are not unique persisted random IDs")
    public: list[dict[str, Any]] = []
    private: list[dict[str, Any]] = []
    schema_counts: Counter[str] = Counter()
    contract_counts: Counter[str] = Counter()
    activation_counts: Counter[str] = Counter()
    geometry_counts: Counter[str] = Counter()
    verified_images: dict[str, tuple[int, int, str]] = {}
    partial_pairwise_public_support: list[dict[str, Any]] = []
    terminal_instruction_residuals: list[dict[str, str]] = []
    for manifest_row in manifest_rows:
        private_path = str(manifest_row["qa_relpath"])
        qa_path = args.dataset_root / private_path
        if sha256_file(qa_path) != str(manifest_row["qa_sha256"]):
            raise ValueError(f"QA file hash mismatch: {private_path}")
        with qa_path.open("rb") as stream:
            question_values = list(ijson.items(stream, "Question"))
        if len(question_values) != 1:
            raise ValueError(f"public question missing: {private_path}")
        question = str(question_values[0])
        if sha256_text(question) != str(manifest_row["question_sha256"]):
            raise ValueError(f"question hash mismatch: {private_path}")
        width = int(manifest_row["image_width_px"])
        height = int(manifest_row["image_height_px"])
        (
            question_1000,
            coordinates_abs,
            coordinates_1000,
            coordinate_repairs,
        ) = convert_question_coordinates(question, width, height)
        question_1000, axis_thresholds = convert_axis_thresholds(
            question_1000, width, height
        )
        raw_boxes = explicit_support_boxes(
            question, coordinates_abs, width, height
        )
        schema = infer_answer_schema(question)
        geometry_encoding = infer_geometry_encoding(question, schema)
        pairwise = pairwise_support_required(question, schema)
        boxes = raw_boxes
        if pairwise and 0 < len(raw_boxes) < 2:
            partial_pairwise_public_support.append(
                {"sample_id": identities[private_path], "parsed_supports": len(raw_boxes)}
            )
            # A partial public pair must never suppress the common support
            # proposal.  The complete public question remains visible to the
            # planner, while the incomplete seed is not privileged.
            boxes = []
        contract = compile_evidence_contract(
            question, schema, boxes, geometry_encoding
        )
        planning_question = strip_answer_instruction(question_1000)
        if contract["support_source"] == "MODEL_PROPOSED_SUPPORT" and (
            match := TERMINAL_OUTPUT_RESIDUAL_RE.search(planning_question)
        ):
            terminal_instruction_residuals.append(
                {"sample_id": identities[private_path], "match": match.group(0)}
            )
        sample_id = identities[private_path]
        image_path = args.dataset_root / str(manifest_row["image_relpath"])
        image_key = str(image_path.resolve())
        if image_key not in verified_images:
            from PIL import Image

            Image.MAX_IMAGE_PIXELS = None
            with Image.open(image_path) as image:
                actual_size = image.size
            actual_hash = sha256_file(image_path)
            verified_images[image_key] = (actual_size[0], actual_size[1], actual_hash)
        actual_width, actual_height, actual_hash = verified_images[image_key]
        if (actual_width, actual_height) != (width, height):
            raise ValueError(f"manifest/image dimensions differ: {private_path}")
        if actual_hash != str(manifest_row["image_sha256"]):
            raise ValueError(f"manifest/image hash differs: {private_path}")
        alias_path = public_image_alias(
            image_path,
            str(manifest_row["image_sha256"]),
            args.public_out.parent / "public_images",
        )
        public.append(
            {
                "schema_version": "map-public-contract-2",
                "sample_id": sample_id,
                "question_original": question,
                "question_source1000": question_1000,
                "question_support_planning": planning_question,
                "image_abs_path": str(alias_path),
                "image_sha256": str(manifest_row["image_sha256"]),
                "source_width_px": width,
                "source_height_px": height,
                "input_coordinate_protocol": "absolute_pixels_declared_in_public_question",
                "model_facing_question_coordinate_protocol": "source_image_base1000",
                "output_coordinate_protocol": "source_image_base1000",
                "coordinates_abs_from_public_question": coordinates_abs,
                "coordinates_source1000_from_public_question": coordinates_1000,
                "axis_thresholds_abs_from_public_question": axis_thresholds,
                "public_coordinate_boundary_repairs": coordinate_repairs,
                "explicit_support_boxes_abs": boxes,
                "evidence_contract": contract,
            }
        )
        # This one-way binding is kept outside every inference-time artifact.
        private.append(
            {
                "sample_id": sample_id,
                "qa_relpath": private_path,
                "task_key": str(manifest_row["task_key"]),
                "split_position": int(manifest_row["split_position"]),
            }
        )
        schema_counts[schema] += 1
        geometry_counts[geometry_encoding] += 1
        for operator in contract["observation_operators"]:
            contract_counts[operator] += 1
        for activation in contract["operator_activations"]:
            activation_counts[activation] += 1

    public_ids = {row["sample_id"] for row in public}
    private_ids = {row["sample_id"] for row in private}
    forbidden = {
        "task_key",
        "task_family",
        "qa_relpath",
        "split",
        "answer",
        "annotation",
        "metric",
        "dataset_version",
        "historical_version",
    }
    leaked = sorted(
        {
            key
            for row in public
            for key in row
            if key.lower() in forbidden
        }
    )
    audit = {
        "schema_version": "map-public-boundary-audit-2",
        "validation_manifest_sha256": manifest_sha256,
        "validation_manifest_hash_locked": True,
        "validation_rows_and_prefix_verified": True,
        "rows": len(public),
        "unique_sample_ids": len(public_ids),
        "schema_counts": dict(sorted(schema_counts.items())),
        "geometry_encoding_counts": dict(sorted(geometry_counts.items())),
        "observation_geometry_encoding_counts": dict(
            sorted(
                Counter(
                    row["evidence_contract"]["observation_geometry_encoding"]
                    for row in public
                ).items()
            )
        ),
        "terminal_geometry_encoding_counts": dict(
            sorted(
                Counter(
                    row["evidence_contract"]["terminal_geometry_encoding"]
                    for row in public
                ).items()
            )
        ),
        "observation_operator_counts": dict(sorted(contract_counts.items())),
        "operator_activation_counts": dict(sorted(activation_counts.items())),
        "public_coordinate_boundary_repair_samples": sum(
            bool(row["public_coordinate_boundary_repairs"]) for row in public
        ),
        "public_coordinate_boundary_repair_records": [
            {
                "sample_id": row["sample_id"],
                "repairs": row["public_coordinate_boundary_repairs"],
            }
            for row in public
            if row["public_coordinate_boundary_repairs"]
        ],
        "public_private_id_sets_equal": public_ids == private_ids,
        "private_fields_in_public_rows": leaked,
        "answer_field_accessed": False,
        "question_read_by_streaming_field_selector": True,
        "full_qa_object_materialized": False,
        "opaque_random_identity_map": str(args.identity_map),
        "opaque_identity_values_unique_random_64hex": True,
        "qa_file_bytes_verified_without_materializing_answer": True,
        "image_bytes_and_dimensions_verified": True,
        "runtime_task_identity_available": False,
        "content_addressed_public_image_aliases": True,
        "partial_pairwise_public_support_replanned": partial_pairwise_public_support,
        "terminal_instruction_residuals": terminal_instruction_residuals,
        "pairwise_support_contract_errors": [
            row["sample_id"]
            for row in public
            if row["evidence_contract"]["support_topology"] == "PAIRWISE_SUPPORT"
            and not (
                (
                    row["evidence_contract"]["support_source"]
                    in {"PUBLIC_EXPLICIT_SUPPORT", "PUBLIC_TARGET_SUPPORT"}
                    and len(row["explicit_support_boxes_abs"]) >= 2
                )
                or (
                    row["evidence_contract"]["support_source"]
                    == "MODEL_PROPOSED_SUPPORT"
                    and not row["explicit_support_boxes_abs"]
                    and row["evidence_contract"]["max_support_points"] >= 2
                )
            )
        ],
        "test_used": False,
    }
    audit["passed"] = (
        len(public) == len(public_ids)
        and public_ids == private_ids
        and not leaked
        and not terminal_instruction_residuals
        and not audit["pairwise_support_contract_errors"]
    )
    write_jsonl(args.public_out, public)
    write_jsonl(args.private_binding_out, private)
    write_json(args.audit_out, audit)
    if not audit["passed"]:
        raise RuntimeError(f"public boundary failed: {audit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
