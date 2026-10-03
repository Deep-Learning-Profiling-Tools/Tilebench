#!/usr/bin/env python3
"""Explicit CSV latency comparisons and conservative stored-ratio checks."""

import argparse
import csv
import json
import math
from decimal import Decimal, InvalidOperation
from pathlib import Path


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _half_unit(value):
    """Absolute half-width implied by the last printed decimal digit."""
    try:
        decimal = Decimal(str(value).strip())
    except (InvalidOperation, ValueError):
        return 0.0
    exponent = decimal.as_tuple().exponent
    return float(Decimal(5).scaleb(exponent - 1)) if exponent < 0 else 0.5


def _ratio_interval(numerator, denominator):
    n = _number(numerator)
    d = _number(denominator)
    if n is None or d is None or n <= 0 or d <= 0:
        return None
    n_error, d_error = _half_unit(numerator), _half_unit(denominator)
    low_denominator = d - d_error
    if low_denominator <= 0:
        return None
    return ((n - n_error) / (d + d_error), (n + n_error) / low_denominator)


def _overlaps(left, right):
    return left[0] <= right[1] and right[0] <= left[1]


def _printed_interval(value):
    center = _number(value)
    if center is None:
        return None
    error = _half_unit(value)
    return (max(0.0, center - error), center + error)


def _identity(row, identity):
    return all(str(row.get(key, "")) == str(value) for key, value in identity.items())


def _latency_like(column):
    lowered = column.lower()
    return (lowered.endswith(("_ms", "_us", "_ns"))
            or "latency" in lowered or "time" in lowered)


def analyze_csv(path, identity, comparisons, stored_ratios=(), candidate_latency_columns=None):
    """Analyze one uniquely selected row.

    Each comparison is {name, numerator_column, denominator_column}. Stored ratio
    checks enumerate candidate ordered pairs and only accept a unique match.
    """
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        rows = list(reader)

    matching = [(index, row) for index, row in enumerate(rows, start=2) if _identity(row, identity)]
    result = {
        "csv": str(path),
        "identity": dict(identity),
        "selected_row": None,
        "status": "ok",
        "comparisons": [],
        "stored_ratio_checks": [],
    }
    if not matching:
        result["status"] = "missing_row"
        return result
    if len(matching) > 1:
        result["status"] = "ambiguous_row"
        result["matching_csv_lines"] = [index for index, _ in matching]
        return result

    line_number, row = matching[0]
    result["selected_row"] = {"csv_line": line_number, "identity": {k: row.get(k) for k in identity}}
    dtype_keys = [key for key in fields if "dtype" in key.lower() or "type" == key.lower()]
    case_keys = [key for key in fields if key not in identity and key not in dtype_keys]
    result["selected_row"]["dtype"] = {key: row.get(key) for key in dtype_keys}
    result["selected_row"]["case"] = {key: row.get(key) for key in case_keys}

    for spec in comparisons:
        name = spec.get("name")
        numerator_column = spec.get("numerator_column")
        denominator_column = spec.get("denominator_column")
        item = {"name": name, "numerator_column": numerator_column,
                "denominator_column": denominator_column, "status": "ok"}
        if not all((name, numerator_column, denominator_column)):
            item["status"] = "invalid_spec"
        elif numerator_column not in fields or denominator_column not in fields:
            item["status"] = "missing_column"
        else:
            numerator, denominator = _number(row.get(numerator_column)), _number(row.get(denominator_column))
            if numerator is None or denominator is None:
                item["status"] = "invalid_value"
            elif numerator <= 0 or denominator <= 0:
                item["status"] = "non_positive_value"
            else:
                item["numerator"] = numerator
                item["denominator"] = denominator
                item["ratio"] = numerator / denominator
        result["comparisons"].append(item)

    candidate_columns = candidate_latency_columns if candidate_latency_columns is not None else [
        key for key in fields if key not in identity and key not in stored_ratios
        and _latency_like(key) and _number(row.get(key)) is not None and _number(row.get(key)) > 0
    ]
    for ratio_column in stored_ratios:
        check = {"stored_ratio_column": ratio_column, "status": "ok", "candidates": []}
        if ratio_column not in fields:
            check["status"] = "missing_column"
            result["stored_ratio_checks"].append(check)
            continue
        if not candidate_columns:
            check["status"] = "no_latency_candidates"
            result["stored_ratio_checks"].append(check)
            continue
        observed = _printed_interval(row.get(ratio_column))
        observed_value = _number(row.get(ratio_column))
        if observed_value is None or observed_value <= 0:
            check["status"] = "invalid_stored_ratio"
            result["stored_ratio_checks"].append(check)
            continue
        matches = []
        for numerator_column in candidate_columns:
            for denominator_column in candidate_columns:
                if numerator_column == denominator_column:
                    continue
                if numerator_column not in fields or denominator_column not in fields:
                    continue
                interval = _ratio_interval(row.get(numerator_column), row.get(denominator_column))
                if interval is not None and _overlaps(interval, observed):
                    matches.append({
                        "numerator_column": numerator_column,
                        "denominator_column": denominator_column,
                        "ratio": _number(row[numerator_column]) / _number(row[denominator_column]),
                        "ratio_interval": list(interval),
                    })
        check["candidates"] = matches
        if not matches:
            check["status"] = "no_match"
        elif len(matches) > 1:
            check["status"] = "ambiguous"
        else:
            check["match"] = matches[0]
        result["stored_ratio_checks"].append(check)

    if any(item["status"] != "ok" for item in result["comparisons"] + result["stored_ratio_checks"]):
        result["status"] = "validation_issues"
    return result


def compare_references(path, identity, target_column, reference_columns, stored_ratios=()):
    """Derive comparisons and the fastest reference; labels never choose the denominator."""
    references = list(dict.fromkeys(reference_columns))
    if not references or target_column in references:
        raise ValueError("Provide at least one reference column, excluding the target")
    specs = [{"name": f"{target_column} / {column}",
              "numerator_column": target_column, "denominator_column": column}
             for column in references]
    result = analyze_csv(path, identity, specs, stored_ratios,
                         [target_column, *references])
    ranking = {"target_column": target_column, "reference_columns": references,
               "status": "unresolved", "fastest_reference_columns": []}
    result["reference_ranking"] = ranking
    if len(result["comparisons"]) != len(references) or any(
        item["status"] != "ok" for item in result["comparisons"]
    ):
        return result
    best = min(item["denominator"] for item in result["comparisons"])
    winners = [item for item in result["comparisons"] if item["denominator"] == best]
    ranking.update({"status": "ok", "fastest_latency": best,
                    "fastest_reference_columns": [item["denominator_column"] for item in winners],
                    "target_over_fastest": winners[0]["ratio"],
                    "precision_note": "Ranking uses printed CSV latency values; equal values remain tied."})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv")
    parser.add_argument("--identity", required=True, help="JSON object selecting one row, e.g. '{\"dtype\":\"fp16\",\"params\":\"N=1024\"}'")
    parser.add_argument("--comparison", action="append", default=[], metavar="NAME:NUMERATOR:DENOMINATOR")
    parser.add_argument("--stored-ratio", action="append", default=[])
    parser.add_argument("--candidate-column", action="append", default=None)
    parser.add_argument("--target-column", help="Derive all target/reference ratios and ranking")
    parser.add_argument("--reference-column", action="append", default=[])
    args = parser.parse_args()
    comparisons = []
    for item in args.comparison:
        parts = item.split(":", 2)
        if len(parts) != 3:
            parser.error(f"invalid --comparison: {item}")
        comparisons.append(dict(zip(("name", "numerator_column", "denominator_column"), parts)))
    try:
        identity = json.loads(args.identity)
        if not isinstance(identity, dict):
            raise ValueError("identity must be a JSON object")
    except (json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))
    if args.target_column or args.reference_column:
        if not args.target_column or not args.reference_column or comparisons or args.candidate_column:
            parser.error("Use target + reference columns without comparison/candidate-column flags")
        try:
            result = compare_references(args.csv, identity, args.target_column,
                                        args.reference_column, args.stored_ratio)
        except ValueError as error:
            parser.error(str(error))
    else:
        result = analyze_csv(args.csv, identity, comparisons, args.stored_ratio, args.candidate_column)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
