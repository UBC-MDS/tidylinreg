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
def empty_model():
    model = LinearModel()
    return model

@pytest.fixture
def test_model():
    model = LinearModel()
    model.params = beta_pred
    model.X = X
    model.y = y_true
    return model

@pytest.fixture
def ref_model():
    model = sm.OLS(y_true, X).fit()
    return model

def test_is_model_fitted(empty_model):
    with pytest.raises(ValueError):
        empty_model.get_ci()

def test_error_empty_x(empty_model):
    empty_model.params = beta_pred
    with pytest.raises(ValueError):
        empty_model.get_ci()

def test_error_empty_y(empty_model):
    empty_model.params = beta_pred
    empty_model.X = X
    with pytest.raises(ValueError):
        empty_model.get_ci()

def test_alpha_is_non_numeric(test_model):
    with pytest.raises(TypeError):
        test_model.get_ci(alpha="hello")        

def test_alpha_is_zero(test_model):
    with pytest.raises(ValueError):
        test_model.get_ci(alpha=0)

def test_alpha_is_one(test_model):
    with pytest.raises(ValueError):
        test_model.get_ci(alpha=1)

def test_alpha_below_zero(test_model):
    with pytest.raises(ValueError):
        test_model.get_ci(alpha=-0.01)

def test_alpha_above_one(test_model):
    with pytest.raises(ValueError):
        test_model.get_ci(alpha=1.01)

def test_ci_values(test_model, ref_model):
    test_model.test_statistic = ref_model.tvalues
    test_model.std_error = ref_model.bse

    alpha = 0.05
    expected_ci = ref_model.conf_int(alpha=alpha)
    test_model.get_ci(alpha=alpha)

    assert np.allclose(test_model.ci, expected_ci)
