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

## shape_query.py
This file lets the Step 4 interface use the KD-tree. In the interface you can choose between the exact search of Step 4 and the ANN search, and set K (the number of results). The results are shown in the same cards. For a shape of the database the descriptors in the index are used. A shape from outside the database is first repaired, resampled and normalised, and then its descriptors are put in the column order of the index. To start the interface:
```python -m src.step_4.user_interface.main```
