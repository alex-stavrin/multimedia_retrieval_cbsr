# 2.3
The python file for resampling is ```resampling.py```  

You can run it:  
```python -m src.step_2._3.resampling -i ./3d_models -o ./resampled_models -pdl -psl -pos```  

- i = defines the 3d models folder
- o = defines the output folder where we will put our resampled models
- pdl = enable debugging messages for decimating (reducing the vertices count)
- psl = enable debugging messages for subdivision (increasing the vertices count)
- pos = enable debugging messages for outliers and their stats

## Triangle Area Distribution Analyzer
The python file for analyzing triangle area evenness before and after resampling is ```triangle_area_analyzer.py```.

You can run it:
```python -m src.step_2._3.triangle_area_analyzer -b ./3d_models_final/3d_models_steps/1_models_reparation -a ./3d_models_final/3d_models_steps/2_models_resampled -o ./3d_models_final/analysis/2_models_resampled/triangle_area_distribution.csv```
