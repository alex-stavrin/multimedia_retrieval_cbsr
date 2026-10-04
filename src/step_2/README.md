# 2.1
File is ```shapes_analyzer.py```  

## How to run
You run it from project root directory. Example:    
```python -m src.step_2.shapes_analyzer -i ./3d_models/Bed/D00031.obj -s True -o ./analysis_output/test.csv -v true```

# 2.2
File is ```database_analyzer.py```

# 2.3
The python file for this is ```resampling_outliers.py```  

You can run it:  
```python -m src.step_2.resampling_outliers -icsv ./analysis_output/shape_analysis.csv -m ./3d_models -o ./resampled_models -pdl -psl -pos```