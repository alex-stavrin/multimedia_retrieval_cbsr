# shapes_analyzer.py

```shapes_analyzer.py``` analyzes 3D `.obj` shapes and/or database directories, extracting the shape class, number of faces, number of vertices, axis-aligned bounding box, and face type (`TRIANGLES`, `QUADS`, `MIXED`, `POLY`), and writes the results to a CSV file. It can also visualize the shapes with their bounding boxes using ```ModelViewer```.

You can run it from the root directory with:  
```python -m src.step_2._1.shapes_analyzer -i ./3d_models/Bed/D00031.obj ./3d_models/Bus/ ./3d_models/AquaticAnimal/ -o ./analysis_output/output.csv -v```  
-i / --input = One or more `.obj` files and/or directories to analyze (directories are searched recursively).  
-o / --output = Output path where the CSV analysis results will be saved. If omitted, results are printed to the console.  
-v / --visual = If present, opens the 3D viewer to visualize the analyzed shapes together with their bounding boxes.
