# DR (Dimensionality Reduction)

## tsne_engine.py
This file reduces the feature vectors of all shapes to 2D with t-SNE (scikit-learn) and saves the points, so the scatterplot does not have to run t-SNE again. It uses the same csv files and the same columns as the KD-tree of `src/step_5/ANN`. The output has one row per shape with the columns `file_path`, `shape_class`, `x` and `y`. You can run it:
```python -m src.step_5.DR.tsne_engine --elementary_csv {path_to_elementary_desciptors_csv} --histogram_csv {path_to_histogram_descriptors_csv} -o {name_of_output_file}```

The default output is `analysis_output/embeddings/tsne.csv`, and the points can be read again with `load_embedding()`.

The options are `--perplexity` (default 30) and `--seed` (default 0, so the result is the same every run). The Euclidean distance on the raw vectors is mostly decided by the groups with the largest values (6 scalars against 320 histogram bins), so by default every group of columns (the scalars together, and each of the five histograms) is first scaled to a total variance of 1. With `--no_balance` the vectors are used as they are.

At the end the program prints how many of the 10 nearest neighbours of a shape have the same class, in the feature space and in the 2D plot. On the 2 483 shapes (perplexity 30) this is 0.260 in the feature space and 0.232 in the plot, and 0.213 in the plot without the balancing. Perplexity 5 to 100 gave values between 0.215 and 0.241, and the trustworthiness of the plot (10 neighbours) was highest at perplexity 30, with 0.956.
