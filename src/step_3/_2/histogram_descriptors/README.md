# How to run

## analyze_and_plot
This calculates the histogram descriptors for the first {number_of_classes} classes and plots them.

```python -m src.step_3._2.histogram_descriptors.analyze_and_plot -i {input_3d_folder} -d {descriptor_to_compute} -s {number_of_samples} -cl {number_of_classes} -c {number_of_columns} -b {number_of_bins} -pdf```
  
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
  
Flag -pdf, if present will save the figure to a .pdf file

## analyze_and_save
This calculates the histogram descriptors and saves them in {output}

```python -m src.step_3._2.histogram_descriptors.analyze_and_save -i {input_3d_folder} -o {output} -s {number_of_samples} -b {number_of_bins}```