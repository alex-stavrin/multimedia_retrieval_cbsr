# Multimedia Retrieval CBSR

## 1. About

Assignement for Multimedia Retrieval  
Website : https://webspace.science.uu.nl/~telea001/MR/Assignment  

## 2. Report

Report : https://www.overleaf.com/project/6aa7ba137686e3f065d4cd14  

## 3. Programming Language

Python 3.14.7

## 4. How to Setup

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

## 5. How to Develop

### Adding Libraries
If you add a new library do  
```py -m pip freeze > requirements.txt```  
So others can install the library too

## 6. Libraries

- vedo
- PySide6

## 7. How to Run the Steps

Every command is run from the project root (the folder that contains `src`)
with the virtual environment activated. Each step has its own instructions:

- Step 1 (viewer): [src/step_1/README.md](src/step_1/README.md)
- Steps 2.1, 2.2, 2.3, the reduced database and 2.5: [src/step_2/README.md](src/step_2/README.md)
- Step 3.1: [src/step_3/README.md](src/step_3/README.md)

For quick tests, first create the reduced database (about 276 shapes), then run
the steps on it instead of the full `3d_models` folder.