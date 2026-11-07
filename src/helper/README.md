# Helper

## prepare_database.py

Prepares the 3D model database through reparation, resampling, normalization, descriptor extraction, and retrieval evaluation with checkpoint resume support.

### Command Format

```bash
python -m src.helper.prepare_database -i {input_database} -o {output_directory} [options]
```

### Arguments

- `-i`, `--input`: (Required) Path to the raw input database folder.
- `-o`, `--output`: Path to the output directory (default: `3d_models_prepared`).
- `-a`, `--analyze`: Run optional shape and reparation analysis steps.
- `--reset_checkpoint`: Delete checkpoint and rerun pipeline from scratch.
- `--reset_last_step`: Rerun the last unfinished/completed step and subsequent steps.
- `--evaluate` / `--no-evaluate`: Toggle retrieval evaluations (kNN, rNN, weighted) (default: `True`).
- `--weights_csv`: Path to weights CSV file for weighted evaluation.
- `--knn_k`: Number of nearest neighbors for kNN evaluation (default: `10`).
- `--rnn_radius`: Radius for rNN evaluation (default: `0.5`).
- `--weighted_k`: Number of nearest neighbors for weighted evaluation (default: `10`).
- `--weighted_config`: Distance metric for weighted evaluation (`emd`, `l2`, `cosine`, default: `emd`).
- `--seed`: Random seed for histogram descriptor sampling (default: `42`).
- `--normalize_by_sum` / `--normalize_by_area`: Normalize histograms by sum (default) or by area.

### Example

```python -m src.helper.prepare_database -i ./3d_models -o ./3d_models_final -a --evaluate --weights_csv ./analysis_output/distances/weights_emd_descriptors --knn_k 6 --rnn_radius 2.5 --weighted_k 6 --normalize_by_sum```