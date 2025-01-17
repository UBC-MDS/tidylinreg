import pandas as pd
import numpy as np
from numpy.linalg import inv, LinAlgError
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
        None
        '''
        self.params = None
        self.param_names = None
        self.X = None
        self.y = None
        self.in_sample_predictions = None
        self.n_samples = None
        self.n_features = None
  
          
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
    
    def get_std_error(self, X):
        '''
        Get the standard error for parameter estimates.

        The standard error for the coefficients in the fitted model are computed and returned.
        Note that model must be fitted first.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input features for calculation of standard error.        

        Returns
        -------
        array-like of shape (n_samples,)
            The calculated standard error values.
        '''
        return
    
    def get_test_statistic(self, X):
        '''
        Get the t-test statistic of parameter estimates to be used 
        in hypothesis testing for statistical significance.

        The t-test statistic for the coefficients in the fitted model are computed and returned.
        Note that model must be fitted first.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input features for calculation of the t-test statistic(s).        

        Returns
        -------
        array-like of shape (n_samples,)
            The calculated t-test statistic values.
        '''
        return
    
    def get_ci(self, type="two-tailed", alpha=0.05):
        '''
        Get the confidence interval obtained from a lower- or upper-tailed or two-tailed
        hypothesis test for the statistical significance of the coefficients in the model.

        The confidence interval(s) for the coefficients in the fitted model are computed and returned.
        Note that model must be fitted first.

        Parameters
        ----------
        alpha : array-like of shape (n_samples, n_features)
            The input features for calculation of the t-test statistic(s).        

        Returns
        -------
        alpha: float, optional
            The significance level used to compute confidence intervals. By default, 0.05 (ie. a 95% C.I).
            If ci=False, does nothing.
        '''
        return
    
    def get_pvalues(self):
        '''
        Compute the significance p-value for each parameter estimate.
        
        p-values are computed parametrically using the t-test. 
        p-values are computed parametrically using the t-test. 
        
        Parameters
        ----------
        None
        
        Returns
        -------
        pd.Dataframe:
            The significance p-values for each parameter in the model.
        
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
        return
    
    def summary(self, **kwargs) -> pd.DataFrame:
        '''
        Provides a summary of the model fit, similar to the output of the R summary() function when computed on
        a fitted `lm` object.
        
        If model has been fitted, a dataframe containing parameter estimates, standard errors, significance
        p-values, and (optional) standard error is returned. As is the case with other 
        methofs, These statistics are computed using bootstrapping if 'method' attribute is set to 'bootstrap'
        during initialization. Note that model must be fitted before summary.
        
        Parameters
        ----------
        **kwargs
            arguments used to specify whether or not confidence intervals are included in summary,
            and their significance level (see LinearModel.get_error_metrics documentation for details).
            
        Returns
        -------
        pd.DataFrame
            A DataFrame containing the summary of the fitted model.
        
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
        >>> model.summary(ci=True, alpha = 0.5)
        '''
        return
        
    
    
    