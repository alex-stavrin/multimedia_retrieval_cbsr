# resampling.py

```resampling.py``` resamples 3D meshes in the database so that every shape has between 5,000 and 10,000 vertices (targeting 7,500 vertices) and an even triangle area distribution, using quadric decimation for high-vertex meshes and adaptive edge/linear/centroid subdivision for low-vertex meshes.

You can run it from the root directory with:  
```python -m src.step_2._3.resampling -i ./3d_models -o ./resampled_models -pdl -psl -pos```  
-i / --input = Input folder or `.obj` file containing the 3D meshes to resample.  
-o / --output = Output folder where the resampled 3D meshes will be saved.  
-pdl / --print_decimate_logs = Enable console logging for decimated meshes (reducing vertex count).  
-psl / --print_subdivision_logs = Enable console logging for subdivided meshes (increasing vertex count).  
-pos / --print_outliers_stats = Enable console logging for meshes that were already within the target range.  
-pfl / --print_failed_logs = Print failed subdivision and decimation logs (enabled by default; disable with `--no_print_failed_logs`).
