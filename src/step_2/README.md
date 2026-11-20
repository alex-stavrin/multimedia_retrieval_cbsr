# reduced_database.py

```reduced_database.py``` copies a small, class-balanced subset of the 3D shape database (by default 4 shapes per class, 276 shapes in total) using a fixed random seed so that pipeline steps can be tested quickly and reproducibly.

You can run it from the root directory with:  
```python -m src.step_2.reduced_database -i ./3d_models -o ./3d_models_reduced -n 4 --seed 42```  
-i / --input = Path to the full 3D shape database root folder.  
-o / --output = Path to the output directory where the reduced database will be written.  
-n / --shapes-per-class = Maximum number of shapes to sample per class (default is 4).  
--seed = Random seed for reproducible shape selection (default is 42).
