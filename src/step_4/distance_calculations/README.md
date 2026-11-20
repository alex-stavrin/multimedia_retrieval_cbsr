# normalize_descriptors.py

```normalize_descriptors.py``` standardizes the numeric elementary descriptor columns in a CSV file using z-score normalization (subtracting the column mean and dividing by the sample standard deviation) so that every scalar feature has zero mean and unit variance.

You can run it from the root directory with:  
```python -m src.step_4.distance_calculations.normalize_descriptors -i ./analysis_output/descriptors/simple_descriptors.csv -o ./analysis_output/descriptors/simple_descriptors_normalized.csv```  
-i / --input = Path to the raw elementary descriptors CSV file.  
-o / --output = Output path for the standardized descriptors CSV file.  
--ignore = Optional list of metadata columns to exclude from standardization (defaults to `file_path`, `was_closed`, `reason`).

# distance_functions.py

```distance_functions.py``` implements the distance metrics used to compare scalar and histogram shape descriptors (`euclidean_distance`, `cosine_distance`, and 1D CDF `earth_movers_distance`), and can be run directly to compute the distance between a query shape and specific comparison shapes for a single descriptor.

You can run it from the root directory with:  
```python -m src.step_4.distance_calculations.distance_functions -i ./analysis_output/descriptors/histogram_descriptors.csv -q AquaticAnimal/m59.obj -s AquaticAnimal/m51.obj Bed/D00031.obj --descriptor A3 --distance_function emd --histogram```  
-i / --input = Path to the CSV file containing the mesh descriptors.  
-q / --query = Query mesh identifier, filename, or relative path.  
-s / --shapes = One or more comparison mesh identifiers, filenames, or relative paths.  
--descriptor = Name of the descriptor to compare (for example `compactness` or `A3`).  
--distance_function = Distance metric to use (`euclidean`, `cosine`, or `emd`).  
--histogram = If present, treats `--descriptor` as a multi-bin histogram family (`A3`, `D1`, `D2`, `D3`, `D4`).  
--visualize = If present, displays the query and comparison meshes side by side in a 3D window.

# calculate_all_pairwise_distances.py

```calculate_all_pairwise_distances.py``` uses ```ShapeDistanceCalculator``` from ```calculate_total_distance.py``` to compute per-descriptor distances across all unique pairs of meshes in the database (with checkpoint resume support), producing the pairwise distance CSV used to derive descriptor weights and distance scales.

You can run it from the root directory with:  
```python -m src.step_4.distance_calculations.calculate_all_pairwise_distances -c emd -i ./analysis_output/descriptors/simple_descriptors_normalized.csv ./analysis_output/descriptors/histogram_descriptors.csv -o ./analysis_output/distances/all_pairwise_distances.csv```  
-c / --config = Distance configuration to use (`emd`, `l2`, `cosine`).  
-i / --csv_paths = Paths to the normalized elementary descriptors CSV and histogram descriptors CSV.  
-o / --output = Output CSV path for the pairwise distances.

# calculate_weights.py

```calculate_weights.py``` reads the pairwise distances CSV produced by ```calculate_all_pairwise_distances.py``` in chunks and computes the per-descriptor pairwise mean, standard deviation, equal weights, and histogram-to-scalar `distance_scale` factors.

You can run it from the root directory with:  
```python -m src.step_4.distance_calculations.calculate_weights --csv_path ./analysis_output/distances/all_pairwise_distances.csv -c emd -o ./analysis_output/distances/weights_emd_hist_sampling_descriptors.csv```  
--csv_path = Path to the pairwise distances CSV file.  
-c / --config = Distance configuration used when computing the pairwise distances (`emd`, `l2`, `cosine`).  
-o / --output = Output CSV path where the descriptor weights and distance scales will be saved.  
--chunksize = Number of CSV rows to process per chunk (default is 500,000).

# calculate_total_distance.py

```calculate_total_distance.py``` has the ```ShapeDistanceCalculator``` class that runs the preprocessing and descriptor extraction pipeline on a query mesh (or looks up precomputed descriptors) and computes the weighted total distance against all database shapes.

You can run it from the root directory with:  
```python -m src.step_4.distance_calculations.calculate_total_distance --csv_paths ./3d_models_final/analysis/4_descriptors/simple_descriptors_normalized.csv ./3d_models_final/analysis/4_descriptors/histogram_descriptors.csv ./analysis_output/distances/weights_emd_hist_sampling_descriptors.csv -q ./3d_models/AquaticAnimal/m59.obj -c emd -o ./query_result.json --visualise```  
--csv_paths = Paths to the normalized elementary descriptors CSV, histogram descriptors CSV, and weights CSV.  
-q / --query = Query mesh identifier, filename, or `.obj` file path.  
-s / --shapes = Optional list of comparison mesh identifiers or database directory path.  
-c / --config = Distance configuration to use (`emd`, `l2`, `cosine`, and `-equal` / `-rec` variants).  
-o / --output = Optional output `.json` file path to save the full distance breakdown.  
--max_top_results_printing = Number of top nearest matches to print in the console (default is 10).  
--visualise = If present, renders the query shape and top retrieved shapes in an interactive 3D window.
