# analyze_and_save.py

```analyze_and_save.py``` uses the functions from ```histogram_descriptors.py``` to compute all 5 shape property histogram descriptors (`A3`, `D1`, `D2`, `D3`, and `D4`) for every `.obj` mesh in a database directory and saves the resulting feature vectors to a CSV file.

You can run it from the root directory with:  
```python -m src.step_3._2.histogram_descriptors.analyze_and_save -i ./3d_models_full_normalized -o ./analysis_output/descriptors/histogram_descriptors.csv -s 100000 -b 64 --seed 42 --normalize_by_sum```  
-i / --input = Input root folder of 3D `.obj` shapes to analyze.  
-o / --output = Output `.csv` file path where the histogram descriptors will be saved.  
-s / --samples = Number of random surface point samples per shape for each descriptor (default is 100,000).  
-b / --bins = Number of bins per histogram descriptor (default is 64).  
-w / --workers = Number of parallel worker processes to use (default is all CPU cores).  
--seed = Random seed for reproducible surface sampling (default is 42).  
--normalize_by_sum = If present, normalizes each histogram so that its bin values sum to 1 (instead of area integral = 1).

# analyze_and_plot.py

```analyze_and_plot.py``` uses the functions from ```histogram_descriptors.py``` to compute a selected histogram descriptor (`A3`, `D1`, `D2`, `D3`, or `D4`) across shapes in the database and plots the overlaid line distributions grouped by class.

You can run it from the root directory with:  
```python -m src.step_3._2.histogram_descriptors.analyze_and_plot -i ./3d_models_full_normalized -d A3 -s 100000 -b 64 -cl 12 -c 4 -pdf```  
-i / --input = Input root folder of 3D `.obj` shapes to analyze.  
-d / --descriptor = Which histogram descriptor to compute and plot (`A3`, `D1`, `D2`, `D3`, or `D4`; default is `A3`).  
-s / --samples = Number of random samples to take per shape (default is 100,000).  
-b / --bins = Number of histogram bins (default is 64).  
-cl / --classes = Number of shape classes to include in the plot grid (default is 12).  
-c / --columns = Number of subplot columns per row (default is 4).  
-ym / --ymax = Optional maximum y-axis value; if omitted, each class subplot scales dynamically.  
--seed = Random seed for reproducible sampling (default is 42).  
--normalize_by_sum = If present, normalizes histograms so their bin values sum to 1.  
-pdf / --pdf = If present, saves the figure to a `.pdf` file.

# histogram_descriptors.py

```histogram_descriptors.py``` implements uniform triangular surface point sampling and the 5 shape distribution histogram descriptors: `calculate_angle_histogram` (`A3`), `calculate_distance_from_barycenter` (`D1`), `calculate_distance_two_vertices` (`D2`), `calculate_area_triangle` (`D3`), and `calculate_volume_tetrahedron` (`D4`).
