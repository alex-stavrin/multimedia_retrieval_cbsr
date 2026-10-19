# 2.3
The python file for this is ```resampling.py```  

You can run it:  
```python -m src.step_2._3.resampling -icsv ./analysis_output/shape_analysis.csv -m ./3d_models -o ./resampled_models -pdl -psl -pos```  

- icsv = defines the input .csv file  
- m = defines the 3d models folder
- o = defines the output folder where we will put our resampled models
- pdl = enable debugging messages for decimating (reducing the vertices count)
- psl = enable debugging messages for subdivision (increasing the vertices count)
- pos = enable debugging messages for outliers and their stats