# About
This folder contains linear systems encountered from machine learning problems
solved by [GeNIOS](https://github.com/tjdiamandis/GeNIOS.jl), which take the form:

```
(A^T diag(w^{(t)}) A) dx = g
```

where `w^{(t)} := l''(Ax^{(t)}-b)`, `l` is the loss function (e.g., logistic
regression), `x^{(t)}` is the current solution, and `g` is some vector (not
saved in this dataset). More details can be found in their
[pre-print](https://arxiv.org/pdf/2310.08333#page=20). 

### Problems
We focus on three problems, which yield linear systems as written above:
- Huber
- Logistic regression.
Since the matrix `A` is constant at every iteration `t`, we only save it
once while we save `w^{(t)}` at every iteration up until convergence
to high accuracy (dual gap <= 1e-16) or 100 iterations, whichever comes
first.

### Format
For each of the three problems, there is a dedicated folder. Within each,
there contains the `A` matrix stored in `Adata.mtx` (MatrixMarket format).
For each of the vectors `w^{(t)}`s seen for iterations `t=1,...,`, we save it
as as a CSV file as `w_t.csv`.

### Images
We tested on random instances (supplied in the code) with `A` matrices of
size 400x200. We ran GeNIOS, saved the `w_t`s, and then plotted the norm
difference between consecutive `w_t`s as png files. The image show the random
instances tend to converge quickly, as `w_t` does not change much.
