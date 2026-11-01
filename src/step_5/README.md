# Step 5: Scalability

## Spatial Search (KD-Tree / Approximate Nearest Neighbors)

The spatial search engine in [`spatial_search.py`](spatial_search.py) builds a `scipy.spatial.KDTree` over the combined 326-dimensional feature vectors (6 standardized elementary descriptors + 320 histogram bins across `A3`, `D1`, `D2`, `D3`, `D4`).

### Usage

#### 1. K-Nearest Neighbors (K-NN) Search
```powershell
python -m src.step_5.spatial_search -q AircraftBuoyant/m1337.obj -k 6 --exclude_self
```

#### 2. Approximate Nearest Neighbors (ANN) with `eps`
```powershell
python -m src.step_5.spatial_search -q AircraftBuoyant/m1337.obj -k 6 --eps 0.1 --exclude_self
```

#### 3. R-Nearest Neighbors (Range Search)
```powershell
python -m src.step_5.spatial_search -q AircraftBuoyant/m1337.obj -r 0.35 --exclude_self
```

#### 4. Interactive 3D Visualisation with Vedo
```powershell
python -m src.step_5.spatial_search -q AircraftBuoyant/m1337.obj -k 6 --exclude_self --visualise --model_dir 3d_models
```
