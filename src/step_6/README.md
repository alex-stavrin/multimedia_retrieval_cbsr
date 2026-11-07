# Step 6

## run_and_evaluate_weighted.py

Runs and evalutes the weighted approach of step_4. Run it:

```python -m src.step_6.run_and_evaluate_weighted --elementary_csv {elementary_csv_file} --histogram_csv {histogram_csv_file} --output_dir {output_directory_for_results} -k {k_nearest_neighbors} --weights_csv {weights_csv_file}```

### Example

```python -m src.step_6.run_and_evaluate_weighted --elementary_csv ./3d_models_final/analysis/4_descriptors/simple_descriptors_normalized.csv --histogram_csv ./3d_models_final/analysis/4_descriptors/histogram_descriptors.csv --output_dir ./evaluation/knn -k 6 --weights_csv analysis_output/distances/weights_emd_descriptors.csv```

## run_and_evaluate_knn.py

Runs and evaluates the KNN approach. Run it :

```python -m src.step_6.run_and_evaluate_knn --elementary_csv {elementary_csv_file} --histogram_csv {histogram_csv_file} --output_dir {output_directory_for_results} -k {k_nearest_neighbors}```

### Example
```python -m src.step_6.run_and_evaluate_knn --elementary_csv ./3d_models_final/analysis/4_descriptors/simple_descriptors_normalized.csv --histogram_csv ./3d_models_final/analysis/4_descriptors/histogram_descriptors.csv --output_dir ./evaluation/knn -k 6```

## run_and_evaluate_rnn.py

Runs and evaluates the RNN approach. Run it : 
```python -m src.step_6.run_and_evaluate_rnn --elementary_csv {elementary_csv_file} --histogram_csv {histogram_csv_file} --output_dir {output_directory_for_results} -r {radius_for_rnn}```

### Example
```python -m src.step_6.run_and_evaluate_rnn --elementary_csv ./3d_models_final/analysis/4_descriptors/simple_descriptors_normalized.csv --histogram_csv ./3d_models_final/analysis/4_descriptors/histogram_descriptors.csv --output_dir ./evaluation/rnn -r 5```