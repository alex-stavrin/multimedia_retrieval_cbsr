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

Every command is run from the project root (the folder that contains `src`) with the virtual environment activated. For quick tests, first create the reduced database (about 276 shapes, see [src/step_2/README.md](src/step_2/README.md)) and run the steps on it instead of the full `3d_models` folder.
