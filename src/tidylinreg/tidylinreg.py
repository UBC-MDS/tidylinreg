import pandas as pd
import numpy as np
from numpy.linalg import inv, LinAlgError
from scipy import stats
from scipy.stats import t
from numbers import Number
import warnings

class LinearModel:
    '''
    A Linear Model class for various regression tasks, implemented in the style of the R lm()
    function and able to perform bootstrapped estimations.
    '''   
    
    def __init__(self):
        '''
        Initialize LinearModel class.
        
        Parameters
        ----------
        None
        
        '''
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
  
          
    def fit(self,X,y):
        '''
        Fits the linear model to the provided data.

        Parameters
        ----------
        X : pd.DataFrame
            A matrix of named explanatory variables. Expecting shape (n_samples, n_features).
        y : pd.Series
            The observed response vector. Expecting shape (n_samples,).

        Returns
        -------
        None
        
        Examples
        --------
        >>> data = pd.DataFrame({
        ...     "Feature1": [1, 2, 3],
        ...     "Feature2": [4, 5, 6],
        ...     "Target": [7, 8, 9]
        ... })
        >>> X = data[["Feature1", "Feature2"]]
        >>> y = data["Target"]
        >>> model = LinearModel()
        >>> model.fit(y, X)
        '''
        # check number of samples
        if len(y) < 2 or len(X) < 2:
            raise ValueError('less than 2 samples in X or y')
        
        # check types of all entries are numeric only
        if (X.dtypes == object).any() or (y.dtype == object):
            raise TypeError('Non-numeric entries in X or y')
        
        # check for missing entries
        if X.isna().any().any() or y.isna().any():
            raise ValueError('Missing entries in X or y')
        
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
        

    def predict(self,X):
        '''
        Predicts the response variable using the given data. 
        
        Note that the model must be fitted before prediction can occur.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input features for prediction.

        Returns
        -------
        array-like of shape (n_samples,)
            The predicted target values.
            
        Examples
        --------
        >>> train_data = pd.DataFrame({
        ...     "Feature1": [1, 2, 3],
        ...     "Feature2": [4, 5, 6],
        ...     "Target": [7, 8, 9]
        ... })
        ...
        >>> X_train = train_data[["Feature1", "Feature2"]]
        >>> y_train = train_data["Target"]
        ...
        >>> model = LinearModel()
        >>> model.fit(y_train, X_test)
        ...
        >>> X_test = pd.DataFrame({
        ...     "Feature1": [0, 1, 2],
        ...     "Feature2": [3, 4, 5],
        ... })
        >>> y_pred = model.predict(test_data)
        '''
        if type(self.params) == type(None): raise ValueError('model has not been fitted')
        
        X_ones = np.hstack([np.ones([self.n_samples,1]),X])
        return X_ones @ self.params
    
    def get_std_error(self):
        '''
        Get the standard error for parameter estimates.

        The standard error for the coefficients in the fitted model are computed and returned.
        Note that model must be fitted first.      

        Returns
        -------
        array-like of shape (n_features,)
            The calculated standard error values.
        '''
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
        '''
        Get the t-test statistic of parameter estimates to be used
        in hypothesis testing for statistical significance.

        The t-test statistic for the coefficients in the fitted model are
        computed and returned. Note that model must be fitted first.

        Parameters
        ----------
        self.params (array-like): The fitted model coefficients.
        self.std_error (array-like): The calculated standard error values.

        Returns
        -------
        array-like: The calculated t-test statistic values.

        Examples
        --------
        >>> data = pd.DataFrame({
        ...     "Feature1": [1, 2, 3],
        ...     "Feature2": [4, 5, 6],
        ...     "Target": [7, 8, 9]
        ... })
        >>> X = data[["Feature1", "Feature2"]]
        >>> y = data["Target"]
        >>> model = LinearModel()
        >>> model.fit(y, X)
        >>> model.get_test_statistic()
        '''

        self.test_statistic = (self.params / self.std_error).values
        return self.test_statistic

    def get_ci(self, alpha=0.05):
        '''
        Get the confidence interval obtained from a two-tailed hypothesis test for the statistical significance of the coefficients in the model.

        The confidence interval(s) for the coefficients in the fitted model are computed and returned.
        Note that model must be fitted first.       

        Parameters
        ----------
        alpha : float, optional
            The significance level used to compute confidence intervals. By default, 0.05 (ie. a 95% C.I).
            If ci=False, does nothing.
        '''
        if self.params is None:
            raise ValueError("The model must be fitted before standard error values can be computed.")
        
        if self.X is None:
            raise ValueError("Train data (X) is not found. Has the model been fitted?")
        
        if self.y is None:
            raise ValueError("Train data (y) is not found. Has the model been fitted?")        
        
        if not isinstance(alpha, Number):
            raise TypeError("`alpha` argument must be a of numeric type that is greater than 0 and smaller than 1")
        
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
        '''
        Compute the significance p-value for each parameter estimate using the
        t-test. The degrees of freedom (df) are calculated as the number of
        observations minus the number of predictors.

        Model should be fitted before p-values can be calculated.

        Parameters
        ----------
        self.n_samples (int): The number of samples in X.
        self.n_features (int): The number of features in X.
        self.test_statistics (array-like): The calculated t-test statistic values.

        Returns
        -------
        array-like: The significance p-values for each parameter in the model.

        Examples
        --------
        >>> data = pd.DataFrame({
        ...     "Feature1": [1, 2, 3],
        ...     "Feature2": [4, 5, 6],
        ...     "Target": [7, 8, 9]
        ... })
        >>> X = data[["Feature1", "Feature2"]]
        >>> y = data["Target"]
        >>> model = LinearModel()
        >>> model.fit(y, X)
        >>> model.get_pvalues()
        '''
        self.df = self.n_samples - (self.n_features + 1)
        if self.df <= 0:
            raise ValueError("Degrees of freedom must be greater than 0.")
        self.pvalues = [2 * (1-stats.t.cdf(np.abs(t), self.df)) for t in self.test_statistic]
        return self.pvalues

    def summary(self, **kwargs) -> pd.DataFrame:
        '''
        Provides a summary of the model fit, similar to the output of the R summary() function when computed on
        a fitted `lm` object.

        Parameters
        ----------
        **kwargs : dict
            Additional arguments:
            - ci : bool, optional
                Whether to include confidence intervals. Default is False.
            - alpha : float, optional
                Significance level for confidence intervals. Must be between 0 and 1. Default is 0.05.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the summary of the fitted model.

        Raises
        ------
        ValueError
            If the model is not fitted or `alpha` is out of range.
        '''
        # Ensure the model is fitted
        if self.params is None:
            raise ValueError("Model must be fitted before generating a summary.")

        # Extract arguments from kwargs with defaults
        ci = bool(kwargs.get("ci", False))
        alpha = kwargs.get("alpha", 0.05)

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