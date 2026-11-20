# main.py

```main.py``` uses the ```EmbeddedViewer``` class from ```viewer.py``` and ```ShapeDistanceCalculator``` from ```src.step_4.distance_calculations.calculate_total_distance``` to run the graphical user interface for selecting a 3D query shape, querying the database, and inspecting the top-`k` retrieved 3D shapes and their distances. It reads `SIMPLE_DESCRIPTORS_CSV`, `HISTOGRAM_DESCRIPTORS_CSV`, and `WEIGHTS_EMD_CSV` from the `.env` file.

You can run it from the root directory with:  
```python -m src.step_4.user_interface.main```

# viewer.py

```viewer.py``` has the ```EmbeddedViewer``` class that embeds interactive VTK/vedo 3D model views inside the Qt user interface.
