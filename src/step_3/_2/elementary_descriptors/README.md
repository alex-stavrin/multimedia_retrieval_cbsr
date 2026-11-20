# descriptor_analyzer.py

```descriptor_analyzer.py``` uses the descriptor functions from ```descriptor_extracting.py``` to compute the 6 elementary 3D shape descriptors (`surface_area`, `compactness`, `rectangularity`, `diameter`, `convexity`, and `eccentricity`, plus `was_closed`) for every `.obj` mesh in a database folder and writes them to a CSV file.

You can run it from the root directory with:  
```python -m src.step_3._2.elementary_descriptors.descriptor_analyzer -i ./3d_models_full_normalized -o ./analysis_output/descriptors/simple_descriptors.csv -w 6```  
-i / --input = Root directory containing the `.obj` meshes to analyze.  
-o / --output = Output CSV path where the calculated descriptors will be saved (default is `shape_descriptors.csv`).  
-w / --workers = Number of parallel worker processes to use (default is all CPU cores).

# descriptor_extracting.py

```descriptor_extracting.py``` contains the functions that calculate the 6 elementary shape descriptors (`calculate_surface_area`, `calculate_compactness`, `calculate_rectangularity`, `calculate_diameter`, `calculate_convexity`, `calculate_eccentricity`) for a 3D mesh, and can be run directly to inspect a single `.obj` file or visualize its PCA oriented bounding box using ```ModelViewer```.

You can run it from the root directory with:  
```python -m src.step_3._2.elementary_descriptors.descriptor_extracting -i ./3d_models_full_normalized/Bed/D00031.obj -v```  
-i / --input = Path to the `.obj` file to compute elementary descriptors for.  
-v / --visualize = If present, opens ```ModelViewer``` to visualize the mesh and its PCA oriented bounding box.
