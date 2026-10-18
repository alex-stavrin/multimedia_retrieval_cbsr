# Multimedia Retrieval CBSR

## About

Assignement for Multimedia Retrieval  
Website : https://webspace.science.uu.nl/~telea001/MR/Assignment  

## Report

Report : https://www.overleaf.com/project/6aa7ba137686e3f065d4cd14  

## Programming Language

Python 3.14.7

## How to Setup

### Windows
- Install python
- Create your virtual environment ```py -m venv .venv```
- Activate your virtual environment ```.\.venv\Scripts\Activate.ps1```
- Download the libraries ```python -m pip install -r requirements.txt```

### 3D Models Database

Database sourced from:
https://github.com/MaxRee94/ShapeDatabase_INFOMR

You can download the 3d models locally by doing :  
```git pull```  
```git submodule update --init --recursive```

## How to Develop

### Adding Libraries
If you add a new library do  
```py -m pip freeze > requirements.txt```  
So others can install the library too

## Libraries

- vedo
- PySide6

## Source Code

Source code is inside the ```./src``` folder. The source is divided into steps from the assignment. Each folder has inside a readme that explains the script used for that step and how to run it.

## How to Run

Every command is run from the project root (the folder that contains `src`)
with the virtual environment activated. Each step is separate, so you can run
only the one you need. The readme in each step's folder has more details.

### Step 1: view a shape

```
python -m src.step_1.viewer_runner ./3d_models/Bird/D00089.obj
```

In the viewer the content browser on the right lists all shapes (a double
click or Enter opens one), the sidebar has the FOV slider and the edge, vertex
and shading options, and the status bar shows the FPS. You can also start
without a shape (`python -m src.step_1.viewer_runner`) and pick one in the
browser. Give several files to see them next to each other, for example
`... a.obj b.obj -d 0.0 0.5 0` (`-d` is the distance between them).

### Step 2.1: analyse the shapes (bounding box, number of faces, ...)

```
python -m src.step_2._1.shapes_analyzer -i ./3d_models -o ./analysis_output/my_analysis.csv
```

Writes one row per shape to a csv file. `-i` can be shapes and/or folders. The
result for the whole database is already in the repo:
`./analysis_output/output.csv`.

### Step 2.2: statistics of one column of that csv

```
python -m src.step_2._2.database_analyzer -i ./analysis_output/output.csv -p n_points
```

Shows a histogram and the outliers of the column given with `-p`.

### Step 2.3: resample the outliers

```
python -m src.step_2._3.resampling -icsv ./analysis_output/output.csv -m ./3d_models -o ./resampled_models
```

It uses one process per CPU core. If it stops with
`OpenBLAS error: Memory allocation failed`, limit numpy to one thread first
(PowerShell: `$env:OPENBLAS_NUM_THREADS = 1`) and run it again.

### Step 2.4: check the resampling

Not finished yet.

### Test database (optional, for quick tests)

```
python -m src.step_2.reduced_database -i ./3d_models -o ./3d_models_reduced
```

A small copy of the database (276 shapes, 4 per class). Use it as `-i` in the
steps below to save time.

### Step 2.5: center and scale

```
python -m src.step_2._5.normalizer -i ./3d_models_reduced -o ./3d_models_normalized
```

### Step 3.1: full normalization (center, align, flip, scale)

```
python -m src.step_3._1.full_normalization -i ./3d_models_reduced -o ./3d_models_full_normalized
```

For 2.5 and 3.1 add `--verify` to also re-check every saved shape.
`276/276` means all 276 shapes passed. For the final results use
`./resampled_models` (the result of 2.3) as `-i`.

### Step 3.2: descriptors

Histogram descriptors (A3, D1, D2, D3, D4), shown as histograms per class:

```
python -m src.step_3._2.histogram_descriptors.analyzer -i ./3d_models_reduced -o ./analysis_output -d A3
```

`-d` picks the descriptor, `-cl` the number of classes shown and `-s` the number
of samples. The readme in `src/step_3/_2/histogram_descriptors` has the details.
The elementary descriptors (area, compactness, ...) in
`src/step_3/_2/elementary_descriptors` are not finished yet.

## Pipeline

3d_models -> (resampled) -> 3d_models_resampled -> (normalization) -> 3d_models_resampled_normalized
