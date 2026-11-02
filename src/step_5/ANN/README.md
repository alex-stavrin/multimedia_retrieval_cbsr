# ANN (Approximate Nearest Neighbors)

## knn_search.py
This file makes the knn data structure and does a knn query on query shapes. You can run it: 
```python -m src.step_5.ANN.knn_search -q {querie_shapes} -k {number_of_neighbors_to_return} --elementary_csv {path_to_elementary_desciptors_csv} --histogram_csv {path_to_histogram_descriptors_csv} --model_dir {3d_database_directory} -o {name_of_output_file}```

## rnn_search.py

## shape_query.py
This file lets the Step 4 interface use the KD-tree. In the interface you can choose between the exact search of Step 4 and the ANN search, and set K (the number of results). The results are shown in the same cards. For a shape of the database the descriptors in the index are used. A shape from outside the database is first repaired, resampled and normalised, and then its descriptors are put in the column order of the index. To start the interface:
```python -m src.step_4.user_interface.main```
