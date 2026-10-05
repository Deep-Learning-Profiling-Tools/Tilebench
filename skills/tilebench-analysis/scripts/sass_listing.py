#!/usr/bin/env python3
"""Parse static SASS instructions from NCU, cuobjdump, or nvdisasm listings."""

import argparse
import json
import re
from collections import Counter
from pathlib import Path


ADDRESS_PREFIX = re.compile(r"^\s*(0x[0-9a-fA-F]+)\s+(.*)$")
COMMENT_ADDRESS_PREFIX = re.compile(r"^\s*/\*\s*([0-9a-fA-F]+)\s*\*/\s*(.*)$")
OPCODE = re.compile(r"^[A-Z][A-Z0-9_]*(?:\.[A-Z0-9_]+)*$")
PREDICATE = re.compile(r"^@(!?)(P[0-7]|PT|UP[0-7]|UPT)$", re.IGNORECASE)
KERNEL_LINE = re.compile(r"^\s*(?:Kernel Name|Function)\s*:?\s*(.*?)\s*$", re.IGNORECASE)


def _instruction_text(rest):
    rest = re.sub(r"/\*[^*]*\*/", " ", rest).strip()
    predicate = None
    predicate_match = re.match(r"^(@!?\s*(?:P[0-7]|PT|UP[0-7]|UPT))\s+", rest, re.IGNORECASE)
    if predicate_match:
        predicate = re.sub(r"\s+", "", predicate_match.group(1)).upper()
        rest = rest[predicate_match.end():]
    # NCU appends launch/source columns; disassemblers append encoded words.
    fields = re.split(r"\s{2,}", rest, maxsplit=1)
    text = fields[0].strip()
    if text.startswith("|"):
        text = text[1:].strip()
    tokens = text.split()
    if tokens and PREDICATE.fullmatch(tokens[0]):
        predicate = tokens.pop(0).upper()
    if not tokens:
        return None
    opcode = tokens[0].rstrip(";")
    if not OPCODE.fullmatch(opcode):
        return None
    return predicate, opcode


def parse_listing(text, strict=False):
    kernels = []
    current = None
    warnings = []
    global_addresses = set()
    unparsed = []

    def begin_kernel(name):
        nonlocal current
        if current is not None and (current["instruction_count"] or current["addresses"]):
            kernels.append(current)
        current = {"name": name or f"kernel_{len(kernels) + 1}", "instruction_count": 0,
                   "nop_count": 0, "opcodes": Counter(), "predicates": Counter(),
                   "addresses": set(), "instructions": []}

    for line_number, line in enumerate(text.splitlines(), 1):
        header = KERNEL_LINE.match(line)
        if header:
            begin_kernel(header.group(1).strip())
            continue

        match = ADDRESS_PREFIX.match(line)
        style = "ncu"
        if not match:
            match = COMMENT_ADDRESS_PREFIX.match(line)
            style = "cuobjdump"
        if match:
            address = int(match.group(1), 16)
            parsed = _instruction_text(match.group(2))
            if parsed is None:
                # Address-only continuations and encoding/header lines are not instruction-like.
                candidate = match.group(2).strip()
                if re.match(r"(?:0x[0-9a-fA-F]+\s*)?$", candidate) or not candidate:
                    continue
                if re.match(r"^[A-Z][A-Z0-9_?.]*\s+", candidate):
                    unparsed.append({"line": line_number, "text": line})
                continue
            if current is None:
                begin_kernel("kernel_1")
            predicate, opcode = parsed
            if address in current["addresses"]:
                warnings.append({"code": "duplicate_address", "kernel": current["name"],
                                 "line": line_number, "address": hex(address)})
            current["addresses"].add(address)
            global_addresses.add((current["name"], address))
            current["instruction_count"] += 1
            current["opcodes"][opcode] += 1
            if predicate:
                current["predicates"][predicate] += 1
            if opcode == "NOP":
                current["nop_count"] += 1
            current["instructions"].append({"line": line_number, "address": hex(address),
                                             "predicate": predicate, "opcode": opcode, "format": style})
            continue

        # Spot address-bearing disassembly rows that resemble an instruction but did not parse.
        if re.search(r"(?:0x[0-9a-fA-F]{4,}|/\*\s*[0-9a-fA-F]{4,}\s*\*/)", line) and re.search(r"\b[A-Z][A-Z0-9_]*(?:\.[A-Z0-9_]+)*\s+", line):
            unparsed.append({"line": line_number, "text": line})

    if current is not None:
        kernels.append(current)
    if unparsed:
        warnings.extend({"code": "unparsed_instruction_like_line", **item} for item in unparsed)
    if len(kernels) > 1:
        warnings.append({"code": "multiple_kernel_scopes", "kernel_count": len(kernels),
                         "kernels": [kernel["name"] for kernel in kernels]})
    for kernel in kernels:
        kernel["opcodes"] = dict(sorted(kernel["opcodes"].items()))
        kernel["predicates"] = dict(sorted(kernel["predicates"].items()))
        kernel["unique_address_count"] = len(kernel["addresses"])
        kernel["addresses"] = sorted(hex(address) for address in kernel["addresses"])
        kernel["non_nop_instruction_count"] = kernel["instruction_count"] - kernel["nop_count"]
    result = {"status": "ok", "kernel_count": len(kernels), "kernels": kernels,
              "warnings": warnings}
    if warnings:
        result["status"] = "warnings"
    if strict and warnings:
        result["status"] = "error"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("listing")
    parser.add_argument("--strict", action="store_true", help="return error status if warnings are found")
    args = parser.parse_args()
    result = parse_listing(Path(args.listing).read_text(errors="replace"), strict=args.strict)
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["status"] == "error":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
