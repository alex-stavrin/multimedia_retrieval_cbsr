# normalizer.py

```normalizer.py``` translates the barycenter of each 3D shape to the world origin `(0, 0, 0)` and scales the shape uniformly so that the longest side of its axis-aligned bounding box equals `1`. When run on a database folder, it also re-reads and verifies all saved `.obj` files.

You can run it from the root directory with:  
```python -m src.step_2._5.normalizer -i ./resampled_models -o ./3d_models_normalized```  
-i / --input = Path to a single `.obj` file or a database root directory to center and scale.  
-o / --output = Output `.obj` file path or output root directory (preserving class subfolders).
