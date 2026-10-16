# 3.2
File is ```descriptor_extracting.py```

## How to run
You run it from project root directory. Example:
```python -m src.step_2._1.shapes_analyzer -i ./3d_models/Bed/D00031.obj ./3d_models/Bus/ ./3d_models/AquaticAnimals/ -s True -o ./analysis_output/output.csv -v```

This would compute and write analysis of the *folders* 3d_models/Bus/,
3d_models/AquaticAnimals and the single *.obj file* ./3d_models/Bed/D00031.obj
to the single *output file* ./analysis_output/output.csv.