# 3.2

## Descriptor Extracting
File is `descriptor_extracting.py`

### How to run
You run it from project root directory. Example:
`python -m src.step_2._1.shapes_analyzer -i ./3d_models/Bed/D00031.obj ./3d_models/Bus/ ./3d_models/AquaticAnimals/ -s True -o ./analysis_output/output.csv -v`

This would compute and write analysis of the _folders_ 3d*models/Bus/,
3d_models/AquaticAnimals and the single *.obj file* ./3d_models/Bed/D00031.obj
to the single \_output file* ./analysis_output/output.csv.



## Descriptor Plotter
The Python file for this is `plot_shape_features.py`. It loads a feature CSV and plots the distributions of shape descriptors across the database. It can generate histograms for each feature or boxplots grouped by class, using the first folder in each mesh's file_path as its class.

### How to run
Run it from the project root directory using `python -m src.step_2._repairment.plot_shape_features -i ./shape_features.csv`. The `-i/--input` argument specifies the input CSV. By default, the plots are displayed interactively. Use `-o/--output` to save the figure to a file, and `--bins` to set the number of bins for histograms (default: 30).

The optional `--by_class` flag generates boxplots grouped by class, while `--closed_only` restricts the analysis to meshes marked as closed in the CSV. To compare descriptors for specific meshes, use `--compare ./class_a/model1.obj ./class_b/model2.obj`, providing at least two OBJ files, but can be virtually any. This mode ignores the other plot options and compares the selected meshes using their descriptors from the CSV.



## Descriptor Analyzer
The Python file for this is `descriptor_analyzer.py`. It calculates 3D shape descriptors for every OBJ mesh in a database and saves the results to a CSV file. The descriptors include surface area, compactness, rectangularity, diameter, convexity, eccentricity, and whether each mesh is closed. The script also records processing errors in the output files.

### How to run
Run it from the project root directory using `python -m src.step_3._2.elementary_descriptors.descriptor_analyzer -i ./3d_models/ -o ./shape_descriptors.csv`. The `-i`/`--input` argument specifies the root directory containing OBJ meshes, while `-o`/`--output` specifies the output CSV path. The output argument is optional and defaults to `shape_descriptors.csv`. The script searches recursively for OBJ files and preserves their relative paths in the CSV. The input meshes are not modified.

The script generates three output files: the main descriptor CSV, a warnings CSV with the suffix `_warnings.csv`, and a summary text file with the suffix `_conclusion.txt`. The main CSV contains the calculated descriptors for each mesh, with any processing errors recorded in the `reason` column. The warnings CSV records warnings associated with each mesh, while the summary file reports the total number of meshes, successful calculations, errors, and captured warnings.