# How to run

## analyze_and_plot
This calculates the histogram descriptors for the first {number_of_classes} classes and plots them.

```python -m src.step_3._2.histogram_descriptors.analyze_and_plot -i {input_3d_folder} -d {descriptor_to_compute} -s {number_of_samples} -cl {number_of_classes} -c {number_of_columns} -b {number_of_bins} -ym {ymax} -pdf --seed {seed}```
  
This script has many arguements. The only ones you really to bother are -i,-o,-d and maybe -cl. The others have good default values.  

descriptor_to_compute has the following options:  
- A3 = angle between 3 random vertices
- D1 = distance between barycenter and random vertex
- D2 = distance between 2 random vertices
- D3 = square root of area of triangle given by 3 random vertices
- D4=  cube root of volume of tetrahedron formed by 4 random vertices  
  
Default is A3  
  
number_of_classes = The number of shapes classes to compute the descriptor for and show them in the plot. Default value is 12  
number_of_columns = The number of columns per row in the plot. Default is 4.  
number_of_bins = The number of bins in the histogram. Default is 64.  
number_of_samples = The number of samples to take for each descriptor. Default is 100,000.  
ymax (-ym / --ymax) = Maximum value for the y-axis (e.g. 0.15). If omitted, each subplot dynamically scales to effectively use the vertical drawing space.  
seed (--seed) = Random seed for reproducible sampling of random vertices. Default is 42.  
  
Flag -pdf, if present will save the figure to a .pdf file

## analyze_and_save
This calculates the histogram descriptors and saves them in {output}

```python -m src.step_3._2.histogram_descriptors.analyze_and_save -i {input_3d_folder} -o {output} -s {number_of_samples} -b {number_of_bins} --seed {seed}```

## compare_shapes (2 Similar Shapes vs. 1 Different Shape)
Plots the 5 histogram descriptors (`A3`, `D1`, `D2`, `D3`, `D4`) and the 3 compared 3D models (`shapes_3d_comparison.pdf`) as individual figures/PDFs for LaTeX Overleaf (or combined with `--combined`).

```bash
# Save all 5 descriptor PDFs + the 3D models PDF (shapes_3d_comparison.pdf):
python -m src.step_3._2.histogram_descriptors.compare_shapes -pdf --no-show

# Generate ONLY the 3D models comparison PDF (shapes_3d_comparison.pdf):
python -m src.step_3._2.histogram_descriptors.compare_shapes -d models -pdf --no-show

# Plot/save a single specific descriptor (e.g. A3):
python -m src.step_3._2.histogram_descriptors.compare_shapes -d A3 -pdf --no-show

# Custom trio of shapes saved as separate PDFs (5 descriptors + 3D models):
python -m src.step_3._2.histogram_descriptors.compare_shapes Guitar/D00374.obj Guitar/D00655.obj Mug/m507.obj -pdf --no-show

# Save all individual PDFs into a specific folder:
python -m src.step_3._2.histogram_descriptors.compare_shapes -pdf --output-dir ./overleaf_figs --no-show
```

