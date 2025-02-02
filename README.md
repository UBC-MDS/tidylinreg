# tidylinreg
![Python 3.13](https://img.shields.io/badge/python-3.13-blue?style=flat&link=https%3A%2F%2Fwww.python.org%2Fdownloads%2Frelease%2Fpython-3130%2F%3Ffeatured_on%3Dpythonbytes)
[![Documentation Status](https://readthedocs.org/projects/tidylinreg/badge/?version=latest)](https://tidylinreg.readthedocs.io/en/latest/)
[![ci-cd](https://github.com/UBC-MDS/tidylinreg/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/UBC-MDS/tidylinreg/actions/workflows/ci-cd.yml)
[![codecov](https://codecov.io/gh/UBC-MDS/tidylinreg/branch/main/graph/badge.svg)](https://codecov.io/gh/UBC-MDS/tidylinreg)
[![Repo Status](https://img.shields.io/badge/repo%20status-Active-brightgreen)](https://github.com/UBC-MDS/tidylinreg/commits/main/)
![PyPI](https://img.shields.io/pypi/v/tidylinreg)


This package provides tools for linear regression in Python,
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
    - Watch out for collinearity! `tidlinreg` will let you know if there is any linear dependence in your data
    before fitting.
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

## Documentation
Detailed documentation for `tidylinreg` can be found [here](https://tidylinreg.readthedocs.io/en/latest/).

## Using `tidylinreg`

Once `tidylinreg` is installed, you can import the `LinearModel` object to begin your regression analysis!

1. **Fitting the model**

    Before anything else, we need to fit the model to our data:

    ```python
    from tidylinreg.tidylinreg import LinearModel
    import pandas as pd

    training_data = pd.read_csv('path/to/your/training_data.csv')
    X_train = training_data.drop(columns='response')
    y_train = training_data['response']

    my_linear_model = LinearModel()
    my_linear_model.fit(X_train,y_train)
    ```

2. **Summary Statistics**

    Once the regression parameters are estimated, we can summarize their errors and significance using the
    `summary` method:

    ```python
    my_linear_model.summary()
    ```

    By default, the confidence intervals will not be included. We can change this by setting the `ci` argument to `True`:

    ```python
    my_linear_model.summary(ci=True)
    ```

    The default significance level is 0.05, giving 95% confidence intervals. We can change this by modifying the `alpha` arguument. For example, if we want wider 99% confidence intervals, we can set `alpha` to 0.01:

    ```python
    my_linear_model.summary(ci=True, alpha=0.01)
    ```

3. **Make Predictions**

    Now we can make predictions using the `predict` method! Lets suppose we have a subset of our data allocated as
    test data. To make predictions, we can do the following:

    ```python
    testing_data = pd.read_csv('path/to/your/testing_data.csv')
    X_test = testing_data.drop(columns='response')
    
    linear_model.predict(X_test)
    ```

## Testing `tidylinreg`

To test the `tidylinreg` package, you will need to install `pytest` in your python environment:

```bash
$ pip install pytest
```

Then, `git clone` this repository and navigate to the root directory. Execute the following command in your terminal:

```bash
$ pytest
```

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
which can be viewed [here](https://github.com/UBC-MDS/tidylinreg/blob/main/LICENSE).

## Credits

`tidylinreg` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).

## References

- R `lm()` - https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/lm
- R `summary.lm()` - https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/summary.lm
- sklearn Linear Models - https://scikit-learn.org/1.5/modules/linear_model.html
- sklearn Ridge - https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.Ridge.html


## Contributors
- Benjamin Frizzell
- Danish Karlin Isa
- Nicholas Varabioff
- Yasmin Hassan
