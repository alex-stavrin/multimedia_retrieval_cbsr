# 2.1
File is ```shapes_analyzer.py```

## How to run
You run it from project root directory. Example:
```python -m src.step_2.shapes_analyzer -i ./3d_models/Bed/D00031.obj ./3d_models/Bus/ ./3d_models/AquaticAnimals/ -s True -o ./analysis_output/output.csv -v```

This would compute and write analysis of the *folders* 3d_models/Bus/,
3d_models/AquaticAnimals and the single *.obj file* ./3d_models/Bed/D00031.obj
to the single *output file* ./analysis_output/output.csv.
# 2.2
File is ```database_analyzer.py```  

## How to run
You run it from project root directory. Example:
```python -m src.step_2.database_analyzer -i input.csv -p "input_csv_column" (-s).```

The -i flag declares the input, this must be a csv file containing data. The -p
flag declares which column of the input csv will get analyzed. If the -s flag
is provided, the produced plot will be saved to a .pdf file instead of being
presented to the user.

# 2.3
The python file for this is ```resampling_outliers.py```  

You can run it:  
```python -m src.step_2.resampling_outliers -icsv ./analysis_output/shape_analysis.csv -m ./3d_models -o ./resampled_models -pdl -psl -pos```

# Reduced database
The python file for this is ```reduced_database.py```  

It copies a small, class-balanced part of the database (4 shapes per class, 276
shapes in total) so the other steps can be tested quickly. The choice is random
but uses a fixed seed, so everyone who runs it on the same ```3d_models``` folder
gets exactly the same shapes and nobody has to share the files.

## How to run
You run it from project root directory. Example:
```python -m src.step_2.reduced_database -i ./3d_models -o ./3d_models_reduced```

The optional flag -n sets the number of shapes per class and --seed the random
seed.

# 2.5
The python file for this is ```normalizer.py```  

It moves the barycenter of every shape to the origin and scales it uniformly so
that the longest side of its bounding box is 1. In the full pipeline of 3.1 the
scaling is done last, see ```src/step_3/README.md```.

## How to run
You run it from project root directory. Examples:

One shape:  
```python -m src.step_2.normalizer -i ./3d_models/Cup/D00035.obj -o ./3d_models_normalized/Cup/D00035.obj```

A whole folder (the class sub-folders are kept in the output):  
```python -m src.step_2.normalizer -i ./3d_models_reduced -o ./3d_models_normalized```

The -i flag is a single *.obj file* or a *folder*, the -o flag is where the
result is written. For a folder it prints a summary and then re-reads the saved
files and recomputes the barycenter and the longest side of every shape. A line
ending in ```276/276``` means all 276 shapes are correct. ```./3d_models_reduced```
takes about 25 seconds, the full ```./3d_models``` about 2 minutes. Use
```./resampled_models``` (the result of 2.3) as input for the final results.
