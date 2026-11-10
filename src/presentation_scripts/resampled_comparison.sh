#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 3d_models/<shape-path>" >&2
    exit 2
fi

original_shape=$1
model_prefix="3d_models/"

if [[ "$original_shape" != "$model_prefix"* ]]; then
    echo "Shape path must start with '3d_models/': $original_shape" >&2
    exit 2
fi

relative_shape=${original_shape#"$model_prefix"}
resampled_shape="3d_models_prepared/3d_models_steps/2_models_resampled/$relative_shape"

QT_QPA_PLATFORM=xcb uv run -m src.step_1.viewer_runner "$original_shape" "$resampled_shape" -d 0 1 1
