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

### Quick start for 2.5 and 3.1

Run these from the project root (the folder that contains `src`) with the virtual environment activated.

1. Make a small test database (276 shapes):

   ```
   python -m src.step_2.reduced_database -i ./3d_models -o ./3d_models_reduced
   ```

2. Step 2.5, center and scale every shape:

   ```
   python -m src.step_2._5.normalizer -i ./3d_models_reduced -o ./3d_models_normalized
   ```

3. Step 3.1, the full normalization (center, align, flip, scale):

   ```
   python -m src.step_3.full_normalization -i ./3d_models_reduced -o ./3d_models_full_normalized
   ```

At the end each command prints checks like `276/276`, which means all shapes passed. More details are in the readme of each step's folder.
