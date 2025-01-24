# tidylinreg

This package provides tools for linear regression in python,
with a similar style to the `lm` and `summary` functions in R.

## Installation
You can install this package by running the following command in your terminal:
```bash
$ pip install tidylinreg
```

## Summary

The `tidylinreg` package fits a linear model to a dataset, and can be used to carry out regression. 
`tidylinreg` computes and returns a list of summary statistics of the fitted linear model, including standard error, confidence intervals, and p-values.

## Functions

- `fit`: Fits the linear model to the provided data.
- `predict`: Predicts the response variable using the given data.
- `get_error_metrics`: Get error metrics for parameter estimates. The standard error and confidence intervals (optional) are computed and returned.
- `get_p_values`: Compute the significance p-value for each parameter estimate.
- `summary`: Provides a summary of the model fit, similar to the output of the R [`summary()`](https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/summary.lm) function when computed on a fitted `lm` object.

## Python Ecosystem

There are existing models for linear regression in Python, such as [`Ridge`](https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.Ridge.html) from the [`sklearn`](https://scikit-learn.org/1.5/index.html) package. The `tidylinreg` package provides similar `fit` and `predict` functionality, with the added functionality to compute statistical metrics about the linear model, including standard error, confidence intervals, and p-values.

Similar to `tidylinreg`, [`statsmodels`](https://www.statsmodels.org/stable/index.html) is a package that can perform statistical tests on different types of models, including ordinary least squares.

## Contributing

Interested in contributing? Check out the [contributing guidelines](https://github.com/UBC-MDS/tidylinreg/blob/main/CONTRIBUTING.md).
Please note that this project is released with a [Code of Conduct](https://github.com/UBC-MDS/tidylinreg/blob/main/CONDUCT.md).
By contributing to this project, you agree to abide by its terms.


## License

`tidylinreg` was created by Benjamin Frizzell, Danish Karlin Isa, Nicholas Varabioff, Yasmin Hassan. It is licensed under the terms of the MIT license,
which can be viewed [here](https://github.com/UBC-MDS/passwordler/blob/main/LICENSE).

## Credits

`tidylinreg` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).

## References

- R `lm()` - https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/lm
- R `summary.lm()` - https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/summary.lm
- sklearn Linear Models - https://scikit-learn.org/1.5/modules/linear_model.html
- sklearn Ridge - https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.Ridge.html


## Contributors

Benjamin Frizzell, Danish Karlin Isa, Nicholas Varabioff, Yasmin Hassan