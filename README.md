# Multimedia Retrieval CBSR

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

### ENV

Some scripts use an .env file. You should make an .env file before running any steps. An example is this
```
SIMPLE_DESCRIPTORS_CSV=3d_models_final/analysis/4_descriptors/simple_descriptors_normalized.csv

HISTOGRAM_DESCRIPTORS_CSV=3d_models_final/analysis/4_descriptors/histogram_descriptors.csv

WEIGHTS_EMD_CSV=analysis_output/distances/weights_emd_hist_sampling_descriptors.csv
```

## Source Code

Source code is inside the ```./src``` folder. The source is divided into steps from the assignment. Each folder has inside a readme that explains the script used for that step and how to run it.