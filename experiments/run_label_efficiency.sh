#!/usr/bin/env bash
set -euo pipefail

ARCH=${1:-}

if [[ -z "${ARCH}" ]]; then
    echo "Usage: $0 <dense|conv>"
    exit 1
fi

if [[ "${ARCH}" != "dense" && "${ARCH}" != "conv" ]]; then
    echo "Unsupported architecture '${ARCH}'. Expected 'dense' or 'conv'."
    exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
EXPERIMENT_ROOT="${ROOT_DIR}/experiments"
LABEL_DIR="${EXPERIMENT_ROOT}/label_sets"
PROGRESS_DIR="${ROOT_DIR}/artifacts/progress"
ARTIFACT_ROOT="${EXPERIMENT_ROOT}/artifacts/label_efficiency"

COUNTS=(10 25 50 100 250 500)
echo "=== Label Efficiency Experiment ==="
echo "Architecture: ${ARCH}"
echo "Label budgets: ${COUNTS[*]}"

if [[ "${ARCH}" == "dense" ]]; then
    ENCODER_TYPE="dense"
    DECODER_TYPE="dense"
    BATCH_SIZE=512
    MAX_EPOCHS=100
    PATIENCE=20
    LEARNING_RATE="0.001"
else
    ENCODER_TYPE="conv"
    DECODER_TYPE="conv"
    BATCH_SIZE=256
    MAX_EPOCHS=80
    PATIENCE=20
    LEARNING_RATE="0.0005"
fi

LR_TAG=${LEARNING_RATE//./p}
RUN_TAG="arch-${ARCH}_bs-${BATCH_SIZE}_lr-${LR_TAG}_pat-${PATIENCE}_epochs-${MAX_EPOCHS}"
RUN_DIR="${ARTIFACT_ROOT}/${RUN_TAG}"
RESULTS_CSV="${RUN_DIR}/results.csv"

mkdir -p "${LABEL_DIR}" "${RUN_DIR}"

printf "run_tag,architecture,num_labels,labels_path,weights_path,history_path\n" > "${RESULTS_CSV}"

TRAIN_ARGS=(
    --encoder-type "${ENCODER_TYPE}"
    --decoder-type "${DECODER_TYPE}"
    --batch-size "${BATCH_SIZE}"
    --max-epochs "${MAX_EPOCHS}"
    --patience "${PATIENCE}"
    --learning-rate "${LEARNING_RATE}"
)

echo "Run tag: ${RUN_TAG}"

for N in "${COUNTS[@]}"; do
    echo ""
    echo "--- Training with ${N} labels ---"

    LABEL_PATH="${LABEL_DIR}/labels_${N}.csv"
    RUN_BASENAME="${RUN_TAG}_labels-${N}"
    WEIGHTS_PATH="${RUN_DIR}/${RUN_BASENAME}.ckpt"
    HISTORY_DEST="${RUN_DIR}/${RUN_BASENAME}_history.csv"

    echo "Generating labels -> ${LABEL_PATH}"
    python "${ROOT_DIR}/experiments/generate_labels.py" \
        --num-labels "${N}" \
        --output "${LABEL_PATH}"

    echo "Starting training -> ${WEIGHTS_PATH}"
    python "${ROOT_DIR}/scripts/train.py" \
        --labels "${LABEL_PATH}" \
        --weights "${WEIGHTS_PATH}" \
        "${TRAIN_ARGS[@]}"

    HISTORY_SRC="${PROGRESS_DIR}/ssvae_history.csv"
    if [[ -f "${HISTORY_SRC}" ]]; then
        cp "${HISTORY_SRC}" "${HISTORY_DEST}"
        echo "Saved training history -> ${HISTORY_DEST}"
    else
        echo "Warning: expected history at ${HISTORY_SRC} but none found." >&2
    fi

    printf "%s,%s,%s,%s,%s,%s\n" \
        "${RUN_TAG}" \
        "${ARCH}" \
        "${N}" \
        "${LABEL_PATH}" \
        "${WEIGHTS_PATH}" \
        "${HISTORY_DEST}" >> "${RESULTS_CSV}"
done

echo ""
echo "Experiment complete. Results logged to ${RESULTS_CSV}"
