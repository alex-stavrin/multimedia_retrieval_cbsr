# ANN (Approximate Nearest Neighbors)

## knn_search.py
This file makes the ANN data structure and does a knn query on query shapes. You can run it: 
```python -m src.step_5.ANN.knn_search -q {querie_shapes} -k {number_of_neighbors_to_return} --elementary_csv {path_to_elementary_desciptors_csv} --histogram_csv {path_to_histogram_descriptors_csv} --model_dir {3d_database_directory} -o {name_of_output_file}```

### Example

```python -m src.step_5.ANN.knn_search -q .\3d_models\Chess\D01049.obj -k 6 --elementary_csv ./3d_models_final/analysis/4_descriptors/simple_descriptors_normalized.csv  --histogram_csv ./3d_models_final/analysis/4_descriptors/histogram_descriptors.csv --model_dir ./3d_models_final/3d_models_steps/3_models_fully_normalized -o result_knn.png```


## rnn_search.py

This file makes the ANN data structure and does a rnn query on query shapes. You can run it:
```python -m src.step_5.ANN.rnn_search -q {querie_shapes} -r {radius_for_rnn} --max_matches {max_matches} --elementary_csv {csv_with_elementary_descriptors} --histogram_csv {csv_with_histogram_descriptors} --model_dir {database_directory} -o {name_of_output_image}```


### Example

```python -m src.step_5.ANN.rnn_search -q .\3d_models\Chess\D01049.obj -r 500 --max_matches 6 --elementary_csv .\3d_models_final\analysis\4_descriptors\simple_descriptors_normalized.csv --histogram_csv .\3d_models_final\analysis\4_descriptors\histogram_descriptors.csv --model_dir .\3d_models_final\3d_models_steps\3_models_fully_normalized\ -o result_rnn.png```


## evaluate_leaf_size.py

Benchmarks KNN retrieval speed and KD-Tree construction time across different KD-Tree `leaf_size` values and saves CSV tables and graphs (`knn_leaf_size_retrieval_time.png`, `knn_leaf_size_detailed_analysis.png`).

```python -m src.step_5.ANN.evaluate_leaf_size --elementary_csv {elementary_csv} --histogram_csv {histogram_csv} -o {output_dir} -k {k}```


## compare_methods_table.py

Generates a 3-row comparison table (Step 4 Custom Weighted Distance, Step 5 kNN, and Step 5 rNN) for each query shape and exports the tables as PDFs (`result_comparison_chess_D01049.pdf`, `result_comparison_door_D01005.pdf`, `result_comparison_human_D00013.pdf`, `result_comparison_human_m356.pdf`).

```python -m src.step_5.ANN.compare_methods_table```
