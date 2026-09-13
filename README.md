# Multimedia Retrieval CBSR

## 1. About

Assignement for Multimedia Retrieval

## 2. Programming Language

Python 3.14.7

## 3. How to Setup

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

## 4. How to run  
```python main.py```  

## 4. How to Develop

### Adding Libraries
If you add a new library do  
```py -m pip freeze > requirements.txt```  
So others can install the library too

## 5. Libraries

- vedo