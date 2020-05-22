# AutoML Utils
While machine learning is facing a [reproducibility crisis](https://youtube.videoken.com/embed/jH0AgVcwIBc), the problem
is exacerbated in the subdiscipline of automated machine learning [[1](https://arxiv.org/abs/1902.07638), 
[2](https://arxiv.org/abs/1912.12522)] where the number of potential hyperparameters and variations in search and 
training regimens can be vast.

When combined with the duplication of code across projects, often with subtle differences in implementation, it can be
challenging to resolve whether changes in performance stem from improved methodology or from changes in configuration.

This repository aims to facilitate these comparisons by providing reference implementations of commonly used components.
It is designed to be minimal and unopinionated to ensure maximum flexibility for the researcher.

**Note**: This work is intentionally being released in an early state of development to enable use in other projects and
guide future efforts based on feedback received. 
