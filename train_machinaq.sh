#!/usr/bin/env bash
# Standalone MachinaQ training driver.
#
# Trains the local PyTorch models (PointNet, through-hole binary classifier,
# the unified multi-task model, and the operation classifier) end to end.
# Pure local computation only — no network calls, no Claude/AI-API usage of
# any kind. Safe to run from a plain terminal, cron, or CI: `./train_machinaq.sh`
#
# Usage:
#   ./train_machinaq.sh [all|pointnet|through-hole|unified|operation-classifier|mfcad24|fusion-seg|primitive-geometry|gnn] [-- extra run_train.py args]
#
# Examples:
#   ./train_machinaq.sh                       # train everything runnable on this machine
#   ./train_machinaq.sh pointnet              # train just the PointNet model
#   ./train_machinaq.sh unified -- --epochs 5 # override epochs for one stage
#
# The GNN stage (AAGNet on MFInstSeg) needs the `dgl` package plus the
# machgnn/dataset/MFInstSeg dataset, which this checkout doesn't ship with.
# It's skipped automatically unless both are present, or run explicitly and
# failed loudly with `gnn`.

set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
ROOT="$(pwd)"

TARGET="${1:-all}"
if [[ "${TARGET}" != "--" && $# -gt 0 ]]; then shift; fi
if [[ "${1:-}" == "--" ]]; then shift; fi
EXTRA_ARGS=("$@")

# ── Resolve interpreter ──────────────────────────────────────────────────────
if [[ -x "${ROOT}/.venv/bin/python" ]]; then
    PY="${ROOT}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY="$(command -v python3)"
else
    echo "error: no python3 interpreter found (looked for ${ROOT}/.venv/bin/python and python3 on PATH)" >&2
    exit 1
fi

if ! "${PY}" -c 'import torch' >/dev/null 2>&1; then
    echo "error: '${PY}' cannot import torch. Install requirements first, e.g.:" >&2
    echo "  ${PY} -m pip install -r requirements.txt torch" >&2
    exit 1
fi

mkdir -p "${ROOT}/outputs"
STAMP="$(date +%Y%m%d_%H%M%S)"
SUMMARY_LOG="${ROOT}/outputs/train_machinaq_${STAMP}.log"

echo "MachinaQ training  |  interpreter: ${PY}"
echo "Target: ${TARGET}"
echo "Summary log: ${SUMMARY_LOG}"
echo "==============================================================" | tee -a "${SUMMARY_LOG}"

GNN_AVAILABLE=0
if "${PY}" -c 'import dgl' >/dev/null 2>&1 && [[ -d "${ROOT}/machgnn/dataset/MFInstSeg" ]]; then
    GNN_AVAILABLE=1
fi

declare -a STAGES=()
case "${TARGET}" in
    all)
        STAGES=(pointnet through-hole unified operation-classifier mfcad24 fusion-seg primitive-geometry)
        if [[ "${GNN_AVAILABLE}" -eq 1 ]]; then
            STAGES+=(gnn)
        else
            echo "note: skipping gnn stage — needs 'dgl' + machgnn/dataset/MFInstSeg (not present here). Run './train_machinaq.sh gnn' to force it and see the actual error." | tee -a "${SUMMARY_LOG}"
        fi
        ;;
    pointnet|through-hole|unified|operation-classifier|mfcad24|fusion-seg|primitive-geometry|gnn)
        STAGES=("${TARGET}")
        ;;
    *)
        echo "error: unknown target '${TARGET}' (expected: all, pointnet, through-hole, unified, operation-classifier, mfcad24, fusion-seg, primitive-geometry, gnn)" >&2
        exit 1
        ;;
esac

declare -A RESULT=()
OVERALL_RC=0

for stage in "${STAGES[@]}"; do
    echo "" | tee -a "${SUMMARY_LOG}"
    echo "── ${stage} ──────────────────────────────────────────────" | tee -a "${SUMMARY_LOG}"

    stage_args=("--model" "${stage}")
    if [[ "${stage}" == "unified" ]]; then
        # Fine-tune from the pointnet + through-hole checkpoints when they exist,
        # so unified doesn't have to relearn feature/hole-type discrimination
        # from scratch.
        if [[ -f "${ROOT}/outputs/machinaq_pointnet.pth" && -f "${ROOT}/outputs/machinaq_through_hole.pth" ]]; then
            stage_args+=("--merge-weights")
        fi
    fi
    stage_args+=("${EXTRA_ARGS[@]}")

    if "${PY}" run_train.py "${stage_args[@]}" 2>&1 | tee -a "${SUMMARY_LOG}"; then
        RESULT["${stage}"]="OK"
    else
        RESULT["${stage}"]="FAILED"
        OVERALL_RC=1
    fi
done

echo "" | tee -a "${SUMMARY_LOG}"
echo "==============================================================" | tee -a "${SUMMARY_LOG}"
echo "Summary:" | tee -a "${SUMMARY_LOG}"
for stage in "${STAGES[@]}"; do
    printf '  %-22s %s\n' "${stage}" "${RESULT[${stage}]}" | tee -a "${SUMMARY_LOG}"
done
echo "Full log: ${SUMMARY_LOG}" | tee -a "${SUMMARY_LOG}"

exit "${OVERALL_RC}"
