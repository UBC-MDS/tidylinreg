from tidylinreg.tidylinreg import LinearModel

import pytest
import numpy as np
import statsmodels.api as sm

# test linear regression equation >> y = X*beta
X = np.vstack(np.arange(-15, 16, 1))
beta_true = np.array([2])

seed = 524      # DO NOT CHANGE SEED
mu = 0
sigma = 0.5
n = X.shape[0]

np.random.seed(seed)
error = np.vstack(np.random.normal(mu, sigma, n))
y_true = X * beta_true + error
beta_pred = np.linalg.inv(X.T @ X) @ X.T @ y_true
y_pred = X * beta_pred

@pytest.fixture
def test_model():
    """
    Initialise empty model for testing.
    """
    model = LinearModel()
    return model

@pytest.fixture
def ref_model():
    """
    Initialise reference model from `statsmodels.OLS` for verification during testing.
    """
    model = sm.OLS(y_true, sm.add_constant(X)).fit()
    return model

def test_is_model_fitted(test_model):
    """
    Test if `LinearModel` is fitted.
    """
    with pytest.raises(ValueError):
        test_model.get_std_error()

def test_empty_x(test_model):
    """
    Test for empty attribute in `LinearModel`.
    """
    test_model.params = beta_pred
    with pytest.raises(ValueError):
        test_model.get_std_error()

def test_empty_y(test_model):
    """
    Test for empty attribute in `LinearModel`.
    """
    test_model.params = beta_pred
    test_model.X = X
    with pytest.raises(ValueError):
        test_model.get_std_error()

def test_calculate_std_error(test_model, ref_model):
    """
    Test for correctness of output in regular cases.
    """
    test_model.params = beta_pred
    test_model.X = np.hstack([np.ones((n,1)),X])
    test_model.y = y_true
    test_model.in_sample_predictions = y_pred
    test_model.n_samples = n
    test_model.n_features = 1
    expected_std_error = ref_model.bse
    test_model.get_std_error()

    assert np.allclose(test_model.std_error, expected_std_error, atol=0.001)

def test_calculate_std_error_zero(test_model):
    """
    Test for correctness of output for edge case when `std_error` is expected to be 0.
    """
    test_model.params = beta_true
    test_model.X = X
    test_model.y = X * beta_true
    test_model.in_sample_predictions = X * beta_true
    test_model.n_samples = n
    test_model.n_features = 1

    expected_std_error = 0
    test_model.get_std_error()
    assert np.allclose(test_model.std_error, expected_std_error, atol=0.001)
