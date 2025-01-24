# tidylinreg
![Python 3.13](https://img.shields.io/badge/python-3.13-blue?style=flat&link=https%3A%2F%2Fwww.python.org%2Fdownloads%2Frelease%2Fpython-3130%2F%3Ffeatured_on%3Dpythonbytes)
![Documentation Status](https://readthedocs.org/projects/tidylinreg/badge/?version=latest)

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
These summary statistics are ouput as a [Pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html). This is advantageous as it allows for fast and convenient manipulation of large regression models,
where, for example, insignificant parameters can easily be filtered out!

## Functions

`tidylinreg` is built around the `LinearModel` object, which offers three useful methods:

- `fit`:
    - Fits the linear model to the provided regressors and response. This is the first step in using the `LinearModel` object; 
    the object must be fitted to the data before anything else!
    - Please be advised that at the current state of development, `fit` only accepts continuous regressors. If your data is categorical,
    first transforming into dummy variables with encoding techniques, such as [One-Hot Encoding](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)
    provided by Scikit-Learn.
    - For convenience, the intercept is automatically included into the regression model. No need to modify your data to accomodate this!
- `predict`:
    - Predict the response using given test regressor data. Remember to fit the model first!
- `summary`:
    - Provides a summary of the model fit, similar to the output of the R [`summary()`](https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/summary.lm) function when computed on a fitted `lm` object.
    - The output includes parameter names, estimates, standard errors, test statistics, and significance p-values as a [Pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html)
    - Additionally, the user can choose to include confidence interval estimates of their parameters, and can specify the significance level.

The user can access specific aspects of the `summary` function using `get_std_error`, `get_test_statistic`, `get_ci`, and `get_pvalues`.
However, we reccommend using `summary` to access these estimates.

## Python Ecosystem

There are existing models for linear regression in Python, such as [`Ridge`](https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.Ridge.html) from the [`sklearn`](https://scikit-learn.org/1.5/index.html) package. 
The `tidylinreg` package provides similar `fit` and `predict` functionality,
with the added functionality to compute statistical metrics about the linear model, including standard error, confidence intervals, and p-values.
Similar to `tidylinreg`, [`statsmodels`](https://www.statsmodels.org/stable/index.html) is a package that can perform statistical tests on different types of models,
including ordinary least squares. The advantage of `tidylinreg` is the usage of Pandas Dataframes as an output, which assists in optimizing workflows and inference.

## Contributing

Interested in contributing? Check out the [Contributing Guidelines](https://github.com/UBC-MDS/tidylinreg/blob/main/CONTRIBUTING.md).
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