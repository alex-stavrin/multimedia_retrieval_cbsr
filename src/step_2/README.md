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
# 2.2
File is ```database_analyzer.py```


## How to run
You run it from project root directory. Example:
```python -m src.step_2._2.database_analyzer -i input.csv -p "input_csv_column" (-s).```

The -i flag declares the input, this must be a csv file containing data. The -p
flag declares which column of the input csv will get analyzed. If the -s flag
is provided, the produced plot will be saved to a .pdf file instead of being
presented to the user.

# 2.3
The python file for this is ```resampling_outliers.py```

You can run it:
```python -m src.step_2.resampling_outliers -icsv ./analysis_output/shape_analysis.csv -m ./3d_models -o ./resampled_models -pdl -psl -pos```
The optional flag -n sets the number of shapes per class and --seed the random
seed.

The instructions for each step of 2 are in the readme of its folder, for example
```src/step_2/_5/README.md``` for 2.5.
