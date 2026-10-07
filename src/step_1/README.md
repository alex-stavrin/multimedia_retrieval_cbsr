# viewer_example.py

```view_example.py``` uses the ```ModelViewer``` class from ```model_viewer.py``` and actually runs the application. See this example to see how you can use the ```ModeViewer``` in your code.

You can run it from the root directory with:  
```python -m src.step_1.viewer_runner ./3d_models/AquaticAnimal/m59.obj ./resampled_models/AquaticAnimal/m59.obj -d 0.0 0.5 0```  
3d_shape_path = the 3Ds shape you want to load and render  
-d = Is the delta you want between the shapes. For example 0.0 0.5 0.0 will result in a delta of 0.5 in the Y axis.

# model_viewer.py

```model_viewer.py``` has the ```ModelViewer``` class that handles model loading and rendering.