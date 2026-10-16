# 2.2
File is ```database_analyzer.py```  

## How to run
You run it from project root directory. Example:
```python -m src.step_2._2.database_analyzer -i input.csv -p "input_csv_column" (-s).```

The -i flag declares the input, this must be a csv file containing data. The -p
flag declares which column of the input csv will get analyzed. If the -s flag
is provided, the produced plot will be saved to a .pdf file instead of being
presented to the user.