# full_normalization.py

```full_normalization.py``` runs the complete 4-step normalization pipeline on a 3D shape or an entire database folder in the order: translation (barycenter to origin), PCA pose alignment (principal axes aligned with `X`, `Y`, `Z`), moment-test flipping, and uniform unit-cube scaling. When run on a database folder, it also re-reads and verifies all saved `.obj` files.

You can run it from the root directory with:  
```python -m src.step_3._1.full_normalization -i ./resampled_models -o ./3d_models_full_normalized -w 6```  
-i / --input = Path to a single `.obj` file or a database root directory to normalize.  
-o / --output = Output `.obj` file path or output root directory (preserving class subfolders).  
-w / -j / --workers = Number of parallel worker processes to use (default is CPU count).
