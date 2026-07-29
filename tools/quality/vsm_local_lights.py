#!/usr/bin/env python3
"""Fail-closed validation for bounded local-light VSM telemetry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SCHEMA = "bloom-vsm-local-lights-v1"


class ValidationError(RuntimeError):
    pass


def state_from(path: Path) -> dict[str, object]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(document, dict) and "renderer_paths" in document:
            state = document["renderer_paths"]["virtual_shadows"]
        else:
            state = document
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValidationError(f"invalid VSM telemetry {path}: {error}") from error
    if not isinstance(state, dict):
        raise ValidationError(f"invalid VSM telemetry state in {path}")
    return state


def integer(mapping: dict[str, object], field: str, source: str) -> int:
    value = mapping.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{source} field {field!r} must be an integer")
    return value


def evaluate(state: dict[str, object], min_submitted: int) -> list[str]:
    failures: list[str] = []
    if state.get("enabled") is not True or state.get("active") is not True:
        failures.append("VSM was not enabled and active")

    local = state.get("local_lights")
    if not isinstance(local, dict):
        raise ValidationError("telemetry has no local_lights object")
    values = {
        field: integer(local, field, "local_lights")
        for field in (
            "submission_limit",
            "admission_limit",
            "faces_per_light",
            "submitted",
            "visible",
            "admitted",
            "active_shaded",
            "visibility_rejected",
            "budget_suppressed",
            "requested_pages",
            "resident_pages",
            "dirty_pages",
            "rendered_pages",
        )
    }
    if values["submitted"] < min_submitted:
        failures.append(
            f"submitted {values['submitted']} local lights; require at least {min_submitted}"
        )
    if values["submitted"] > values["submission_limit"]:
        failures.append("submitted local lights exceed the public ceiling")
    if values["visible"] + values["visibility_rejected"] != values["submitted"]:
        failures.append("visibility accounting does not sum to submitted lights")
    if values["admitted"] + values["budget_suppressed"] != values["visible"]:
        failures.append("admission accounting does not sum to visible lights")
    if values["admitted"] > values["admission_limit"]:
        failures.append("admitted lights exceed the fixed light budget")
    expected_requested = values["admitted"] * values["faces_per_light"]
    if values["requested_pages"] != expected_requested:
        failures.append("requested pages are not admitted lights times cube faces")
    if not 0 <= values["active_shaded"] <= values["admitted"]:
        failures.append("active shaded lights exceed admitted lights")
    if not 0 <= values["dirty_pages"] <= values["resident_pages"] <= expected_requested:
        failures.append("local residency is outside the admitted page footprint")
    if local.get("shared_page_budget") is not True:
        failures.append("local pages are not reported as using the shared page budget")
    if local.get("fallback") != "suppress-direct-contribution":
        failures.append("local-light fallback is not fail-closed")

    capacity = integer(state, "physical_capacity", "VSM telemetry")
    resident = integer(state, "resident", "VSM telemetry")
    render_budget = integer(state, "render_budget", "VSM telemetry")
    rendered = integer(state, "rendered", "VSM telemetry")
    if resident > capacity:
        failures.append("shared VSM residency exceeds physical capacity")
    if rendered > render_budget:
        failures.append("same-frame VSM rendering exceeds the hard page budget")
    if values["rendered_pages"] > render_budget:
        failures.append("same-frame local rendering exceeds the hard page budget")

    rows = state.get("per_light_cost")
    if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
        raise ValidationError("telemetry has no per_light_cost rows")
    if rows[0].get("kind") != "directional":
        failures.append("first per-light row is not the directional light")
    point_rows = [row for row in rows[1:] if isinstance(row, dict) and row.get("kind") == "point"]
    if len(point_rows) != values["submitted"]:
        failures.append("point-light cost row count differs from submitted lights")
        return failures

    indices: set[int] = set()
    row_totals = {
        "requested_pages": 0,
        "resident_pages": 0,
        "dirty_pages": 0,
        "rendered_pages": 0,
    }
    active_rows = 0
    for row in point_rows:
        light = integer(row, "light", "point-light cost")
        if light in indices:
            failures.append(f"duplicate point-light cost row {light}")
        indices.add(light)
        state_name = row.get("state")
        if state_name not in {"active", "admitted-pending", "suppressed"}:
            failures.append(f"point light {light} has an invalid state")
        row_values = {
            field: integer(row, field, f"point light {light}")
            for field in row_totals
        }
        for field, value in row_values.items():
            row_totals[field] += value
        if row_values["requested_pages"] not in {0, values["faces_per_light"]}:
            failures.append(f"point light {light} has a partial page request")
        if state_name == "active":
            active_rows += 1
            if (
                row_values["requested_pages"] != values["faces_per_light"]
                or row_values["resident_pages"] != values["faces_per_light"]
                or row_values["dirty_pages"] != 0
            ):
                failures.append(f"active point light {light} is not fully resident and clean")
        elif state_name == "suppressed" and row_values["requested_pages"] != 0:
            failures.append(f"suppressed point light {light} consumed page requests")
        if integer(row, "render_budget_pages", f"point light {light}") != render_budget:
            failures.append(f"point light {light} reports a different page budget")

    if active_rows != values["active_shaded"]:
        failures.append("active point-light rows differ from active_shaded")
    for field, expected in (
        ("requested_pages", values["requested_pages"]),
        ("rendered_pages", values["rendered_pages"]),
    ):
        if row_totals[field] != expected:
            failures.append(f"point-light row total for {field} differs from local telemetry")
    # Stale pages from a previously admitted light can remain in the shared
    # LRU pool, so row residency may exceed current local residency. It may
    # never be smaller because every current owner has a submitted row.
    if row_totals["resident_pages"] < values["resident_pages"]:
        failures.append("point-light rows omit resident local pages")
    return failures


def validate(path: Path, min_submitted: int) -> dict[str, object]:
    state = state_from(path)
    failures = evaluate(state, min_submitted)
    local = state.get("local_lights")
    return {
        "schema": SCHEMA,
        "telemetry": str(path),
        "minimum_submitted": min_submitted,
        "observed": local,
        "failures": failures,
        "passed": not failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--min-submitted", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = validate(args.telemetry, args.min_submitted)
    except ValidationError as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for failure in result["failures"]:
        print(f"FAIL: {failure}")
    if result["failures"]:
        return 1
    print("PASS: local VSM submission, visibility, residency, and page cost are bounded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
