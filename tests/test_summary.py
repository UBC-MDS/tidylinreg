import pytest
import numpy as np
import pandas as pd
import statsmodels.api as sm
from tidylinreg.tidylinreg import LinearModel

@pytest.fixture
def linear_model():
    """Return a LinearModel instance."""
    return LinearModel()

@pytest.fixture
def example_data():
    """Generate sample data with a linear relationship and noise."""
    np.random.seed(524)
    X = pd.DataFrame({'x': [-2, -1, 0, 1, 2]})
    y = 3 * X.squeeze() + 2 + np.random.normal(0, 0.5, X.shape[0])
    return X, y

def test_summary_no_fit(linear_model):
    """Ensure summary() raises an error if model is not fitted."""
    with pytest.raises(ValueError):
        linear_model.summary()

def test_summary_basic(linear_model, example_data):
    """Verify summary() returns a DataFrame with expected columns."""
    X, y = example_data
    linear_model.fit(X, y)
    summary_df = linear_model.summary()
    
    assert isinstance(summary_df, pd.DataFrame)
    assert all(col in summary_df.columns for col in 
               ["Parameter", "Estimate", "Std. Error", "T-Statistic", "P-Value"])

def test_summary_with_ci(linear_model, example_data):
    """Ensure summary() includes confidence intervals when requested."""
    X, y = example_data
    linear_model.fit(X, y)
    summary_df = linear_model.summary(ci=True, alpha=0.05)
    
    assert isinstance(summary_df, pd.DataFrame)
    assert all(col in summary_df.columns for col in 
               ["Parameter", "Estimate", "Std. Error", "T-Statistic", "P-Value", "CI Lower", "CI Upper"])

def test_summary_output_without_ci(example_data):
    """Check that summary() excludes confidence intervals when ci=False."""
    X, y = example_data
    model = LinearModel()
    model.fit(X, y)
    model.get_std_error()
    model.get_test_statistic()
    summary = model.summary(ci=False)
    
    assert isinstance(summary, pd.DataFrame)
    assert "CI Lower" not in summary.columns
    assert "CI Upper" not in summary.columns

def test_summary_invalid_alpha(linear_model, example_data):
    """Ensure summary() raises errors for invalid alpha values."""
    X, y = example_data
    linear_model.fit(X, y)

    with pytest.raises(TypeError):
        linear_model.summary(ci=True, alpha="hello")
    with pytest.raises(ValueError):
        linear_model.summary(ci=True, alpha=0)
    with pytest.raises(ValueError):
        linear_model.summary(ci=True, alpha=1)
    with pytest.raises(ValueError):
        linear_model.summary(ci=True, alpha=-0.1)
    with pytest.raises(ValueError):
        linear_model.summary(ci=True, alpha=1.1)

def test_summary_values(linear_model, example_data):
    """Verify summary() estimates match those from statsmodels."""
    X, y = example_data
    linear_model.fit(X, y)
    summary_df = linear_model.summary(ci=True, alpha=0.05)

    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()

    expected_params = model.params.values
    expected_pvalues = model.pvalues

    assert np.allclose(summary_df["Estimate"].values, expected_params, atol=0.1)
