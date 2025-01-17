class LinearModel:
    '''
    A Linear Model class for various regression tasks, implemented in the style of the R lm()
    function and able to perform bootstrapped estimations.
    '''   
    
    def __init__(self, method: str = 'parametric'):
        '''
        Initialize LinearModel class.
        
        Parameters
        ----------
        method : str, optional
            One of ('parametric','bootstrap'). If 'bootstrapped', parameter estimates, standard errors, etc. will be computed using bootstrapping
            from the fitted data. if 'parametric', classical parametric estimates are computed.
        '''
        self.__method = method
        self.params = None
        self.param_names = None
        self.X = None
        self.y = None
  
          
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
        return
    
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
        return
    
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
        import numpy as np

        x = self.X
        y_true = self.y
        y_pred = self.predict(x)

        mean_sq_error = np.mean((y_true - y_pred) ** 2)

        x_bar = np.mean(x, axis=0)
        sum_sq_deviation_x = np.sum((x - x_bar) ** 2, axis=0)

        std_error = np.sqrt(mean_sq_error / sum_sq_deviation_x)

        return std_error
    
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
        type : str
            The type of hypothesis testing conducted. By default, "two-tailed".
            Possible types: ["lower", "upper", "two-tailed"]

        alpha : float, optional
            The significance level used to compute confidence intervals. By default, 0.05 (ie. a 95% C.I).
            If ci=False, does nothing.
        '''
        import numpy as np
        from scipy.stats import t

        x = self.X
        test_statistic = self.get_test_statistic(x)
        n, p = x.shape
        df = n - p

        std_error = self.get_std_error()
        t_critical = t.ppf(1 - alpha / 2, df)

        match type:
            case "two-tailed":
                lower_ci = test_statistic - (std_error * t_critical)
                upper_ci = test_statistic + (std_error * t_critical)

            case "lower":
                lower_ci = test_statistic - (std_error * t_critical)
                upper_ci = np.inf

            case "upper":
                lower_ci = np.inf
                upper_ci = test_statistic + (std_error * t_critical)

        return
    
    def get_pvalues(self):
        '''
        Compute the significance p-value for each parameter estimate.
        
        If method is set to 'parametric', p-values are computed using the t-test. If 'bootstrap', 
        p-values are computed using empirical bootstrapped sample distribution. Model should
        be fitted before p-values can be calculated.
        
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
        
    
    
    