# 2.5
The python file for this is ```normalizer.py```  

It moves the barycenter of every shape to the origin and scales it uniformly so
that the longest side of its bounding box is 1. In the full pipeline of 3.1 the
scaling is done last, see ```src/step_3/README.md```.

## How to run
You run it from project root directory. Examples:

One shape:  
```python -m src.step_2._5.normalizer -i ./3d_models/Cup/D00035.obj -o ./3d_models_normalized/Cup/D00035.obj```

A whole folder (the class sub-folders are kept in the output):  
```python -m src.step_2._5.normalizer -i ./3d_models_reduced -o ./3d_models_normalized```

The -i flag is a single *.obj file* or a *folder*, the -o flag is where the
result is written. For a folder it prints a summary and then re-reads the saved
files and recomputes the barycenter and the longest side of every shape. A line
ending in ```276/276``` means all 276 shapes are correct. ```./3d_models_reduced```
takes about 25 seconds, the full ```./3d_models``` about 2 minutes. Use
```./resampled_models``` (the result of 2.3) as input for the final results.
