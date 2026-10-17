# Reduced database
The python file for this is ```reduced_database.py```  

It copies a small, class-balanced part of the database (4 shapes per class, 276
shapes in total) so the other steps can be tested quickly. The choice is random
but uses a fixed seed, so everyone who runs it on the same ```3d_models``` folder
gets exactly the same shapes and nobody has to share the files.

## How to run
You run it from project root directory. Example:
```python -m src.step_2.reduced_database -i ./3d_models -o ./3d_models_reduced```

This would compute and write analysis of the *folders* 3d_models/Bus/,
3d_models/AquaticAnimals and the single *.obj file* ./3d_models/Bed/D00031.obj
to the single *output file* ./analysis_output/output.csv.