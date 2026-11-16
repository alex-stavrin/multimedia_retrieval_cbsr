# Step 6

## run_and_evaluate_all.py

Runs and evaluates all retrieval methods (`knn`, `rnn`, `weighted`), compares all metrics and execution/retrieval times (`init_time_s`, `retrieval_time_s`, `mean_query_time_ms`, `total_time_s`) side by side in a single table (`all_methods_comparison_table.csv`), computes per-metric and average ranks using `mean_query_time_ms` for timing (`all_methods_ranking_table.csv`), and plots all methods together on a combined ROC curve (`all_methods_roc_comparison.png`), metrics bar chart (`all_methods_metrics_comparison.png`), and mean per-query retrieval time comparison chart (`all_methods_time_comparison.png` and `all_methods_time_comparison.pdf`). Pass `--pdf` / `-pdf` to also export PDF versions of the ROC and metrics comparison plots.

```python -m src.step_6.run_and_evaluate_all --elementary_csv {elementary_csv_file} --histogram_csv {histogram_csv_file} --weights_csv {weights_csv_file} --output_dir {output_directory_for_results} --knn_k {k} --rnn_radius {radius} --weighted_k {k}```

### Example (Run all methods + all 3 weighted distance configs)

```python -m src.step_6.run_and_evaluate_all --elementary_csv ./3d_models_final/analysis/4_descriptors/simple_descriptors_normalized.csv --histogram_csv ./3d_models_final/analysis/4_descriptors/histogram_descriptors.csv --weights_csv analysis_output/distances/weights_emd_hist_sampling_descriptors.csv --output_dir ./evaluation --weighted_configs emd l2 cosine```

## run_and_evaluate_weighted.py

Runs and evaluates the weighted approach of step_4 (including ROC & AUROC). Run it:

```python -m src.step_6.run_and_evaluate_weighted --elementary_csv {elementary_csv_file} --histogram_csv {histogram_csv_file} --output_dir {output_directory_for_results} -k {k_nearest_neighbors} --weights_csv {weights_csv_file}```

### Example

```python -m src.step_6.run_and_evaluate_weighted --elementary_csv ./3d_models_final/analysis/4_descriptors/simple_descriptors_normalized.csv --histogram_csv ./3d_models_final/analysis/4_descriptors/histogram_descriptors.csv --output_dir ./evaluation/weighted -k 6 --weights_csv analysis_output/distances/weights_emd_hist_sampling_descriptors.csv```

## run_and_evaluate_knn.py

Runs and evaluates the KNN approach (including ROC & AUROC). Run it:

```python -m src.step_6.run_and_evaluate_knn --elementary_csv {elementary_csv_file} --histogram_csv {histogram_csv_file} --output_dir {output_directory_for_results} -k {k_nearest_neighbors}```

### Example
```python -m src.step_6.run_and_evaluate_knn --elementary_csv ./3d_models_final/analysis/4_descriptors/simple_descriptors_normalized.csv --histogram_csv ./3d_models_final/analysis/4_descriptors/histogram_descriptors.csv --output_dir ./evaluation/knn -k 6```

## run_and_evaluate_rnn.py

Runs and evaluates the RNN approach (including ROC & AUROC). Run it:
```python -m src.step_6.run_and_evaluate_rnn --elementary_csv {elementary_csv_file} --histogram_csv {histogram_csv_file} --output_dir {output_directory_for_results} -r {radius_for_rnn}```

### Example
```python -m src.step_6.run_and_evaluate_rnn --elementary_csv ./3d_models_final/analysis/4_descriptors/simple_descriptors_normalized.csv --histogram_csv ./3d_models_final/analysis/4_descriptors/histogram_descriptors.csv --output_dir ./evaluation/rnn -r 0.85```
