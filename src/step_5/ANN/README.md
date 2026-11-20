# knn_search.py

```knn_search.py``` uses the ```SpatialSearchEngine``` class from ```spatial_search_engine.py``` to build the KD-Tree index over the database descriptors and run a k-nearest-neighbor (`kNN`) search for one or more query shapes, rendering the retrieved shapes and their distances into an output image.

You can run it from the root directory with:  
```python -m src.step_5.ANN.knn_search -q ./3d_models/Chess/D01049.obj -k 6 --elementary_csv ./3d_models_final/analysis/4_descriptors/simple_descriptors_normalized.csv --histogram_csv ./3d_models_final/analysis/4_descriptors/histogram_descriptors.csv --model_dir ./3d_models_final/3d_models_steps/3_models_fully_normalized -o result_knn.png```  
-q / --queries = One or more query `.obj` file paths or database shape identifiers.  
-k = Number of nearest neighbors to return per query (default is 6).  
--eps = Approximation factor for the KD-Tree search (`0.0` is exact search; default is `0.0`).  
--p = Minkowski distance norm order (`2.0` is Euclidean distance; default is `2.0`).  
--elementary_csv = Path to the normalized elementary descriptors CSV file.  
--histogram_csv = Path to the histogram descriptors CSV file.  
--model_dir = Path to the 3D models directory used to render thumbnails (default is `3d_models`).  
-o / --output = Output file path for the rendered results grid image (default is `knn_result_table.png`).

# rnn_search.py

```rnn_search.py``` uses the ```SpatialSearchEngine``` class from ```spatial_search_engine.py``` to build the KD-Tree index over the database descriptors and run a range nearest-neighbor (`rNN`) search within a given distance radius `r` for one or more query shapes, rendering the retrieved matches into an output image.

You can run it from the root directory with:  
```python -m src.step_5.ANN.rnn_search -q ./3d_models/Chess/D01049.obj -r 0.85 --max_matches 6 --elementary_csv ./3d_models_final/analysis/4_descriptors/simple_descriptors_normalized.csv --histogram_csv ./3d_models_final/analysis/4_descriptors/histogram_descriptors.csv --model_dir ./3d_models_final/3d_models_steps/3_models_fully_normalized -o result_rnn.png```  
-q / --queries = One or more query `.obj` file paths or database shape identifiers.  
-r / --radius = Search radius threshold in the balanced feature space.  
--max_matches = Maximum number of retrieved matches to display per query row (default is 6).  
--eps = Approximation factor for the KD-Tree search (`0.0` is exact search; default is `0.0`).  
--p = Minkowski distance norm order (`2.0` is Euclidean distance; default is `2.0`).  
--elementary_csv = Path to the normalized elementary descriptors CSV file.  
--histogram_csv = Path to the histogram descriptors CSV file.  
--model_dir = Path to the 3D models directory used to render thumbnails.  
-o / --output = Output file path for the rendered results grid image (default is `rnn_table.png`).

# spatial_search_engine.py

```spatial_search_engine.py``` has the ```SpatialSearchEngine``` class that loads and merges the elementary and histogram descriptor CSVs, balances the variance across feature groups, constructs the `scipy.spatial.KDTree` index, and handles `query_knn` and `query_range` (`rNN`) queries (including on-the-fly preprocessing and descriptor extraction for out-of-database query meshes).
