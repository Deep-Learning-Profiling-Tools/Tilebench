"""pytest configuration for tilebench tests.

Usage
-----
Run all operator tests:
    PYTHONPATH=. pytest tests/test_metrics.py -v

Run tests for a specific operator only:
    PYTHONPATH=. pytest tests/test_metrics.py -v --operator mul2
    PYTHONPATH=. pytest tests/test_metrics.py -v --operator destindex

Run tests for multiple operators:
    PYTHONPATH=. pytest tests/test_metrics.py -v --operator mul2 --operator destindex

Without --operator, all registered operator test classes run.
"""
from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--operator",
        action="append",
        default=[],
        metavar="NAME",
        help="Only run tests for the given operator (repeatable). "
             "E.g. --operator mul2 --operator destindex",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    selected = config.getoption("--operator")
    if not selected:
        return

    # Operator test classes follow the naming convention Test<OperatorName>Metrics
    # (case-insensitive match against --operator value).
    selected_lower = {op.lower() for op in selected}

    skip_marker = pytest.mark.skip(reason="operator not in --operator list")
    for item in items:
        cls = item.cls
        if cls is None:
            continue
        cls_name = cls.__name__.lower()  # e.g. "testmul2metrics"
        # Check if any selected operator name appears in the class name
        if not any(op in cls_name for op in selected_lower):
            # Always keep non-operator test classes (utility / guard-rail tests)
            if "metrics" in cls_name:
                item.add_marker(skip_marker)
