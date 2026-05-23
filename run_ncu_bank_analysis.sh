#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Batch export and analyze Nsight Compute reports for TileBench
# ============================================================
#
# Expected layout:
#   ./ncu_report/<operator>/*.ncu-rep
#
# Outputs:
#   ./ncu_bank_outputs/
#     raw_csv/
#     source_csv/
#     details_csv/
#     analysis/
#       ncu_bank_conflict_summary.csv
#       ncu_bank_conflict_summary_enriched.csv
#       paper_figures/*.pdf
#       paper_figures/*.png
#
# Usage:
#   bash run_ncu_bank_analysis.sh
#
# Optional:
#   NCU_REPORT_DIR=ncu_report OUT_DIR=ncu_bank_outputs bash run_ncu_bank_analysis.sh
# ============================================================

ROOT_DIR="$(pwd)"
NCU_REPORT_DIR="${NCU_REPORT_DIR:-${ROOT_DIR}/ncu_report}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/ncu_bank_outputs}"

RAW_CSV_DIR="${OUT_DIR}/raw_csv"
SOURCE_CSV_DIR="${OUT_DIR}/source_csv"
DETAILS_CSV_DIR="${OUT_DIR}/details_csv"
ANALYSIS_DIR="${OUT_DIR}/analysis"
FIG_DIR="${ANALYSIS_DIR}/paper_figures"

ANALYZE_SCRIPT="${ROOT_DIR}/scripts/analyze_ncu_bank_conflicts.py"
PAPER_FIG_SCRIPT="${ROOT_DIR}/scripts/make_ncu_paper_figures.py"

mkdir -p "${RAW_CSV_DIR}" "${SOURCE_CSV_DIR}" "${DETAILS_CSV_DIR}" "${ANALYSIS_DIR}" "${FIG_DIR}"

echo "============================================================"
echo "[TileBench NCU bank-conflict analysis]"
echo "Root dir:        ${ROOT_DIR}"
echo "NCU report dir:  ${NCU_REPORT_DIR}"
echo "Output dir:      ${OUT_DIR}"
echo "============================================================"

if ! command -v ncu >/dev/null 2>&1; then
  echo "[ERROR] ncu command not found. Please load Nsight Compute CLI into PATH."
  echo "Example:"
  echo "  export PATH=/usr/local/cuda/bin:\$PATH"
  echo "or locate ncu under Nsight Compute installation."
  exit 1
fi

if [[ ! -d "${NCU_REPORT_DIR}" ]]; then
  echo "[ERROR] NCU report directory not found: ${NCU_REPORT_DIR}"
  exit 1
fi

if [[ ! -f "${ANALYZE_SCRIPT}" ]]; then
  echo "[ERROR] Missing analyzer script: ${ANALYZE_SCRIPT}"
  exit 1
fi

# ------------------------------------------------------------
# Step 1. Export .ncu-rep reports to CSV pages.
# ------------------------------------------------------------

echo
echo ">>> Step 1: Exporting .ncu-rep files to CSV"

REPORT_COUNT=0
EXPORT_FAIL_COUNT=0

while IFS= read -r -d '' rep; do
  REPORT_COUNT=$((REPORT_COUNT + 1))

  # Operator is parent directory name.
  op="$(basename "$(dirname "${rep}")")"

  # File stem without .ncu-rep.
  stem="$(basename "${rep}" .ncu-rep)"

  # Output filename encodes operator + report stem.
  # Example: matmul_fp32_fp16_fp8__triton_fp32.raw.csv
  out_base="${op}__${stem}"

  raw_out="${RAW_CSV_DIR}/${out_base}.raw.csv"
  source_out="${SOURCE_CSV_DIR}/${out_base}.source.csv"
  details_out="${DETAILS_CSV_DIR}/${out_base}.details.csv"

  echo "  [${REPORT_COUNT}] ${op}/${stem}.ncu-rep"

  # Raw metrics: main input to batch analyzer.
  if ! ncu --import "${rep}" --page raw --csv > "${raw_out}" 2> "${raw_out}.stderr"; then
    echo "    [WARN] raw export failed: ${rep}"
    EXPORT_FAIL_COUNT=$((EXPORT_FAIL_COUNT + 1))
    continue
  fi

  # Source page: useful for source-line attribution; may fail if source info absent.
  if ! ncu --import "${rep}" --page source --csv > "${source_out}" 2> "${source_out}.stderr"; then
    echo "    [WARN] source export failed; continuing"
    rm -f "${source_out}"
  fi

  # Details page: useful for manual inspection; may be larger / version-dependent.
  if ! ncu --import "${rep}" --page details --csv > "${details_out}" 2> "${details_out}.stderr"; then
    echo "    [WARN] details export failed; continuing"
    rm -f "${details_out}"
  fi

done < <(find "${NCU_REPORT_DIR}" -type f -name "*.ncu-rep" -print0 | sort -z)

echo
echo "Exported reports: ${REPORT_COUNT}"
echo "Export failures:  ${EXPORT_FAIL_COUNT}"

if [[ "${REPORT_COUNT}" -eq 0 ]]; then
  echo "[ERROR] No .ncu-rep files found under ${NCU_REPORT_DIR}"
  exit 1
fi

# ------------------------------------------------------------
# Step 2. Run bank-conflict analyzer on raw CSV files.
# ------------------------------------------------------------

echo
echo ">>> Step 2: Running bank-conflict analyzer"

python3 "${ANALYZE_SCRIPT}" \
  --raw-csv-dir "${RAW_CSV_DIR}" \
  --out-dir "${ANALYSIS_DIR}" \
  --pattern "*.raw.csv"

SUMMARY_CSV="${ANALYSIS_DIR}/ncu_bank_conflict_summary.csv"

if [[ ! -f "${SUMMARY_CSV}" ]]; then
  echo "[ERROR] Expected summary not found: ${SUMMARY_CSV}"
  exit 1
fi

# ------------------------------------------------------------
# Step 3. Generate paper-ready figures.
# ------------------------------------------------------------

echo
echo ">>> Step 3: Generating paper-ready figures"

if [[ ! -f "${PAPER_FIG_SCRIPT}" ]]; then
  echo "[ERROR] Missing paper figure script: ${PAPER_FIG_SCRIPT}"
  exit 1
fi

python3 "${PAPER_FIG_SCRIPT}" \
  --summary-csv "${SUMMARY_CSV}" \
  --out-dir "${FIG_DIR}"

echo
echo "============================================================"
echo "Done."
echo "Summary CSV:"
echo "  ${SUMMARY_CSV}"
echo "Paper figures:"
echo "  ${FIG_DIR}"
echo "============================================================"

echo
echo "Generated files:"
find "${ANALYSIS_DIR}" -maxdepth 3 -type f \( -name "*.csv" -o -name "*.pdf" -o -name "*.png" -o -name "*.tex" \) | sort