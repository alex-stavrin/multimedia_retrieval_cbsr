#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 --config <l2|mixed|cosine|l2-equal|mixed-equal|cosine-equal|l2-rec|emd-rec|cosine-rec> --query <query-mesh-path> [-o <output-path>]" >&2
}

config=""
query_path=""
output_path="test.json"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -c|--config)
            config="$2"
            shift 2
            ;;
        -q|--query)
            query_path="$2"
            shift 2
            ;;
        -o|--output)
            if [[ $# -lt 2 ]]; then
                echo "Missing output path after $1" >&2
                usage
                exit 2
            fi
            output_path="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 2
            ;;
    esac
done

if [[ -z "$config" || -z "$query_path" ]]; then
    echo "Both --config and --query are required." >&2
    usage
    exit 2
fi

case "$config" in
    l2) weights_csv="3d_models_prepared/analysis/L2_weights_descriptors.csv" ;;
    l2-equal) weights_csv="analysis_output/distances/weights_equal.csv" ;;
    mixed) weights_csv="3d_models_prepared/analysis/weights_EMD_descriptors.csv" ;;
    mixed-equal) weights_csv="analysis_output/distances/weights_equal.csv" ;;
    cosine) weights_csv="3d_models_prepared/analysis/weights_cosine_descriptors.csv" ;;
    cosine-equal) weights_csv="analysis_output/distances/weights_equal.csv" ;;
    l2-rec) weights_csv="analysis_output/distances/L2_recommended_weights.csv" ;;
    emd-rec) weights_csv="analysis_output/distances/EMD_recommended_weights.csv" ;;
    cosine-rec) weights_csv="analysis_output/distances/cosine_recommended_weights.csv" ;;
    *) echo "Unknown config: $config" >&2; usage; exit 2 ;;
esac

if [[ ! -f "$weights_csv" ]]; then
    echo "Weights CSV does not exist: $weights_csv" >&2
    echo "Generate it with evaluate_retrieval.py --weights-output $weights_csv" >&2
    exit 1
fi

uv run -m src.step_4.distance_calculations.calculate_total_distance \
    --config "$config" \
    --csv_paths \
    "$weights_csv" \
    3d_models_prepared/analysis/histogram_descriptors.csv \
    3d_models_prepared/analysis/simple_descriptors_normalized.csv \
    -q "$query_path" \
    -s 3d_models_prepared/3d_models_steps/3_models_fully_normalized/ \
    -o "$output_path"
