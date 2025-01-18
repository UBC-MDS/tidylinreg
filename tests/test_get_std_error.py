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
y = X * beta_true
y_true = y + error
beta_pred = np.linalg.inv(X.T @ X) @ X.T @ y_true
y_pred = X * beta_pred

@pytest.fixture
def test_model():
    model = LinearModel()
    return model

@pytest.fixture
def ref_model():
    model = sm.OLS(y_true, X).fit()
    return model

def test_is_model_fitted(test_model):
    with pytest.raises(ValueError):
        test_model.get_std_error()

def test_empty_x(test_model):
    test_model.params = beta_pred
    with pytest.raises(ValueError):
        test_model.get_std_error()

def test_empty_y(test_model):
    test_model.params = beta_pred
    test_model.X = X
    with pytest.raises(ValueError):
        test_model.get_std_error()

def test_calculate_std_error(test_model, ref_model):
    test_model.params = beta_pred
    test_model.X = X
    test_model.y = y_true
    test_model.in_sample_predictions = y_pred
    
    expected_std_error = ref_model.bse
    test_model.get_std_error()

    assert np.allclose(test_model.std_error, expected_std_error, atol=0.001)

def test_calculate_std_error_zero(test_model):
    test_model.params = beta_true
    test_model.X = X
    test_model.y = X * beta_true
    test_model.in_sample_predictions = X * beta_true

    expected_std_error = 0
    test_model.get_std_error()
    assert np.allclose(test_model.std_error, expected_std_error, atol=0.001)

def test_features_without_variation(test_model):
    test_model.params = beta_true
    test_model.X = np.vstack(np.ones(10))
    test_model.y = X * beta_true
    test_model.in_sample_predictions = X * beta_true

    with pytest.raises(ValueError):
        test_model.get_std_error()
