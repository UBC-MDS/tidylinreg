import pandas as pd
import numpy as np
from numpy.linalg import inv, LinAlgError
from scipy import stats
from scipy.stats import t
from numbers import Number
import warnings

class LinearModel:
    """
    Ordinary least-squares linear regression for continuous predictors.

    LinearModel is a scaled-down Python implementation of stats::lm in R.
    It can be used to carry out ordinary least-squares regression with
    continuous predictors only, as well as hypothesis testing for the
    statistical significance of the coefficients of the predictors.
    """
    def __init__(self):
        self.initialize()
        return
    
    def initialize(self):
        """
        Initialize the components of the model.
        """
        self.params = None
        self.param_names = None
        self.X = None
        self.y = None
        self.in_sample_predictions = None

        self.n_samples = None
        self.n_features = None
        
        self.std_error = None
        self.test_statistic = None
        self.ci = None
        self.pvalues = None
  
          
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fits the linear model to the provided data.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            A matrix of named explanatory variables.
        y : pd.Series of shape (n_samples,) 
            The observed response vector.

        Raises
        ------
        ValueError
            If there are less than 2 rows in `X` and/or `y`.
        TypeError
            If there are non-numeric entries in `X` and/or `y`.
        ValueError
            If any cell in `X` and/or `y` is empty.
        ValueError
            If `X` and `y` do not have matching dimensions.
        ValueError
            If collinearity is detected in `X`.

        Returns
        -------
        self : `LinearModel` object
            Fitted regression model.

        Examples
        --------
        >>> from tidylinreg.tidylinreg import LinearModel
        >>> import pandas as pd
        >>> data = pd.DataFrame({
        ...     "Feature1": [1, 2, 3],
        ...     "Feature2": [4, 5, 6],
        ...     "Target": [7, 8, 9]
        ... })
        >>> X = data[["Feature1", "Feature2"]]
        >>> y = data["Target"]
        >>> model = LinearModel()
        >>> model.fit(X, y)
        """
        # check number of samples
        if len(y) < 2 or len(X) < 2:
            raise ValueError('less than 2 samples in X or y')
        
        # check types of all entries are numeric only
        if (X.dtypes == object).any() or (y.dtype == object):
            raise TypeError('Non-numeric entries in X or y')
        
        # check for missing entries in X AND y
        if X.isna().any().any() and y.isna().any():
            raise ValueError('Missing entries in X and y')
        
        # check for missing entries in y
        if y.isna().any():
            raise ValueError('Missing entries in y')
        
        # check for missing entries in X
        if X.isna().any().any():
            raise ValueError('Missing entries in X')
        
        # check shape of X and y matches
        if len(y.shape) != 1 or len(X) != y.size:
            raise ValueError('incorrect or mismatching sizes between X and y')
        
        # get number of samples and number of features
        self.n_samples, self.n_features = X.shape
        
        # add ones to X for intercept, and estimate parameters
        # also check for collinear X
        try:
            self.X = np.hstack([np.ones([self.n_samples,1]),X])
            self.y = y
            params = inv(self.X.T @ self.X) @ self.X.T @ self.y
        except LinAlgError:
            raise ValueError('Collinear columns in X')
        
        # get parameter estimates
        self.params = pd.Series(params)
        
        # get parameter names
        self.param_names = ['(Intercept)'] + X.columns.to_list()
        self.params.index = self.param_names
        
        # get in-sample predictions
        self.in_sample_predictions = self.predict(X)
        

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts the response values from a given matrix. 

        The model must be fitted before prediction can occur.

        Parameters
        ----------
        X : pd.DataFrame of shape (n_samples, n_features)
            The input features for prediction.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            The predicted target values.      

        Raises
        ------
        ValueError
            If model is not fitted (`fit()` has not been called).
        
        Examples
        --------
        >>> X_test = pd.DataFrame({
        ...     "Feature1": [0, 1, 2],
        ...     "Feature2": [3, 4, 5],
        ... })
        >>> y_pred = model.predict(test_data)  
        """
        if type(self.params) == type(None): raise ValueError('model has not been fitted')
        
        X_ones = np.hstack([np.ones([len(X),1]),X])
        return X_ones @ self.params
    
    def get_std_error(self):
        """
        Compute the standard error for parameter estimates.

        The model must be fitted before standard error can be calculated.

        Raises
        ------
        ValueError
            If model is not fitted (`fit()` has not been called).

        Returns
        -------
        None

        Notes
        -----
        Results can be accessed by calling `summary()`.
        """
        if self.params is None:
            raise ValueError("The model must be fitted before standard error values can be computed.")
        
        if self.X is None:
            raise ValueError("Train data (X) is not found. Has the model been fitted?")
        
        if self.y is None:
            raise ValueError("Train data (y) is not found. Has the model been fitted?")

        x = self.X
        y_true = self.y
        y_pred = self.in_sample_predictions

        RSS = np.sum((y_true - y_pred) ** 2)/(self.n_samples - self.n_features - 1)

        sum_sq_deviation_x = np.linalg.inv(x.T @ x).diagonal()
        self.std_error = np.sqrt(RSS * sum_sq_deviation_x)

        return

    def get_test_statistic(self):
        """
        Compute the test statistic of the parameter estimates 
        for hypothesis testing.

        The model must be fitted before test statistic(s) 
        can be calculated.

        Raises
        ------
        ValueError
            If model is not fitted (`fit()` has not been called).

        Returns
        -------
        None

        Notes
        -----
        Results can be accessed by calling `summary()`.
        """
        self.test_statistic = (self.params / self.std_error).values

        return

    def get_ci(self, alpha: float=0.05):
        """
        Get the confidence interval of the parameter estimates.

        The confidence interval of the parameter estimates are
        obtained to determine the statistical significance of the
        parameter estimates by conducting a two-tailed hypothesis 
        test with significance level = `alpha`.

        The model must be fitted before confidence interval can 
        be calculated.

        Parameters
        ----------
        alpha : float, optional
            The significance level used to compute confidence intervals,
            by default 0.05

        Raises
        ------
        ValueError
            If model is not fitted (`fit()` has not been called).
        TypeError
            If `alpha` provided is not of numeric type.
        ValueError
            If `alpha` provided does not fall in the range of (0, 1).

        Returns
        -------
        None

        Notes
        -----
        Results can be accessed by calling `summary()`.
        """
        if self.params is None:
            raise ValueError("The model must be fitted before standard error values can be computed.")
        
        if self.X is None:
            raise ValueError("Train data (X) is not found. Has the model been fitted?")
        
        if self.y is None:
            raise ValueError("Train data (y) is not found. Has the model been fitted?")        
        
        if not isinstance(alpha, Number):
            raise TypeError("`alpha` argument must be of numeric type that is greater than 0 and smaller than 1")
        
        if not ((alpha > 0) and (alpha < 1)):
            raise ValueError("`alpha` argument must be a of numeric type that is greater than 0 and smaller than 1")

        x = self.X
        betas = self.params
        n, p = x.shape
        df = n - p

        std_error = self.std_error
        t_critical = t.ppf(1 - alpha / 2, df)
        margin_of_error = std_error * t_critical

        self.ci = np.zeros((p, 2))
        self.ci[:, 0] = betas - margin_of_error
        self.ci[:, 1] = betas + margin_of_error

        return

    def get_pvalues(self):
        """
        Compute the p-value for each parameter estimate
        obtained from conducting a t-test with n-p degrees
        of freedom, where n is n_samples and p is n_features
        in `X` (refer to `fit()` method).
        
        The model must be fitted before confidence interval can be calaculated.

        Returns
        -------
        None

        Notes
        -----
        Results can be accessed by calling `summary()`.    
        """
        self.df = self.n_samples - (self.n_features + 1)
        if self.df <= 0:
            raise ValueError("Degrees of freedom must be greater than 0.")
        self.pvalues = [2 * (1-stats.t.cdf(np.abs(t), self.df)) for t in self.test_statistic]
        return self.pvalues

    def summary(self, ci : bool=False, alpha : float=0.05) -> pd.DataFrame:
        """
        Summarizes the fit of the linear regression model.

        This method provides an overview of the fitted regression model, 
        including parameter estimates, standard errors, test statistics, 
        and p-values. If requested, confidence intervals for the parameter 
        estimates are also included.

        A small p-value (typically ≤ 0.05) indicates strong evidence against 
        the null hypothesis, suggesting that the coefficient is statistically 
        significant.

        The model must be fitted before confidence interval can be calculated.
        

        Parameters
        ----------
        ci : bool, optional
            Whether to include confidence intervals in the summary, 
            by default False
        alpha : float, optional
            Significance level for confidence intervals,
            by default 0.05

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the summary of the fitted model.

        Raises
        ------
        ValueError
            If model is not fitted (`fit()` has not been called).
        TypeError
            If `alpha` provided is not of numeric type.
        ValueError
            If `alpha` provided does not fall in the range of (0, 1).

        Examples
        --------
        >>> from tidylinreg.tidylinreg import LinearModel
        >>> import pandas as pd
        >>> data = pd.DataFrame({
        ...     "Feature1": [1, 2, 3],
        ...     "Feature2": [4, 5, 6],
        ...     "Target": [7, 8, 9]
        ... })
        >>> X = data[["Feature1", "Feature2"]]
        >>> y = data["Target"]
        >>> model = LinearModel()
        >>> model.fit(X, y)
        >>> regression_summary = model.summary()
        """
        # Ensure the model is fitted
        if self.params is None:
            raise ValueError("Model must be fitted before generating a summary.")
        
        if not isinstance(alpha, Number):
            raise TypeError("`alpha` argument must be of numeric type that is greater than 0 and smaller than 1")

        # Validate alpha for confidence intervals
        if ci and (alpha <= 0 or alpha >= 1):
            raise ValueError("Alpha must be between 0 and 1.")

        # Compute standard errors, test statistics, and p-values if not already done
        if self.std_error is None:
            self.get_std_error()
        if self.test_statistic is None:
            self.get_test_statistic()
        if self.pvalues is None:
            self.get_pvalues()

        # Create summary DataFrame
        summary_df = pd.DataFrame({
            "Parameter": self.param_names,
            "Estimate": self.params.values,
            "Std. Error": self.std_error,
            "T-Statistic": self.test_statistic,
            "P-Value": self.pvalues,
        })

        # Add confidence intervals if requested
        if ci:
            if self.ci is None:
                self.get_ci(alpha=alpha)
            summary_df["CI Lower"] = self.ci[:, 0]
            summary_df["CI Upper"] = self.ci[:, 1]

        return summary_df
