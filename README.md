barycenter_ensemble
==============================

A project on using Optimal Transport to make model ensembles

Project Organization
--------------------

```
.
├── AUTHORS.md
├── LICENSE
├── README.md
├── bin                <- Compiled model code (not tracked by git)
├── config             <- Configuration files, e.g., for doxygen or for your model if needed
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
├── docs               <- Documentation, e.g., doxygen or scientific papers (not tracked by git)
├── notebooks          <- Ipython or R notebooks
├── reports            <- For a manuscript source, e.g., LaTeX, Markdown, etc., or any project reports
│   └── figures        <- Figures for the manuscript or reports
└── src                <- Source code for this project
    ├── data           <- scripts and programs to process data
    ├── external       <- Any external source code, e.g., pull other git projects, or external libraries
    ├── models         <- Source code for your own model
    ├── tools          <- Any helper scripts go here
    └── visualization  <- Scripts for visualisation of your results, e.g., matplotlib, ggplot2 related.
```

Plan
-----------------

* Closed form Barycenter, compare this with the standard model averaging
    1. Barycenter of two GP models with biased priors (single input)
        - Prior with biased mean
        - Prior with biased variance or covariance
        - GP with different covariance functions? Show that barycentering GPs achieves the same as using the average covariance function? (this is a little harder to do)
    2. Barycenter of two graphical models with conjugacy structure biased in their own way
        (difficult, requires creating properly biased PGMs and showing closed forms for these)

* Using stoch wass barycenter, and extending this to non-closed form models that are biased (harder).
