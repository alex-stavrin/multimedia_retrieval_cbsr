# repair_wireframe.py

```repair_wireframe.py``` repairs and standardizes 3D meshes in the database by triangulating faces, merging duplicate vertices, removing degenerate/duplicate triangles, closing holes (via hole filling and boundary-loop triangle fans), and fixing triangle winding and outward normals.

You can run it from the root directory with:  
```python -m src.step_2._3.repairment.repair_wireframe -i ./3d_models -o ./3d_models_repaired -w 6```  
-i / --input = Path to a single `.obj` file or the root directory of the 3D shape database.  
-o / --output = Output file or root directory where the repaired 3D meshes and `report.csv` will be saved.  
-s / --steps = Optional list of repair steps to apply in order (`preprocess`, `deduplicate`, `close_holes`, `correct_winding`). By default all four steps are run.  
-w / --workers = Number of parallel worker processes to use (default is 6).
