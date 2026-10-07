# 2.1
File is ```shapes_analyzer.py```

## How to run
You run it from project root directory. Example:
```python -m src.step_2._1.shapes_analyzer -i ./3d_models/Bed/D00031.obj ./3d_models/Bus/ ./3d_models/AquaticAnimals/ -s True -o ./analysis_output/output.csv -v```

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
