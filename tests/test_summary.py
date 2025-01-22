import pytest
import numpy as np
import pandas as pd
import statsmodels.api as sm
from tidylinreg.tidylinreg import LinearModel


@pytest.fixture
def linear_model():
    """Fixture for initializing a LinearModel instance."""
    return LinearModel()


@pytest.fixture
def example_data():
    """Fixture for creating example data."""
    np.random.seed(524)
    X = pd.DataFrame({'x': [-2, -1, 0, 1, 2]})
    y = 3 * X.squeeze() + 2 + np.random.normal(0, 0.5, X.shape[0])
    return X, y


def test_summary_no_fit(linear_model):
    """Test that summary raises an error if the model is not fitted."""
    with pytest.raises(ValueError):
        linear_model.summary()


def test_summary_basic(linear_model, example_data):
    """Test the basic functionality of the summary method."""
    X, y = example_data
    linear_model.fit(X, y)
    summary_df = linear_model.summary()
    
    assert isinstance(summary_df, pd.DataFrame), "Summary output should be a DataFrame."
    assert all(
        col in summary_df.columns
        for col in ["Parameter", "Estimate", "Std. Error", "T-Statistic", "P-Value"]
    ), "Summary should contain the expected columns."


def test_summary_with_ci(linear_model, example_data):
    """Test summary output when confidence intervals are requested."""
    X, y = example_data
    linear_model.fit(X, y)
    summary_df = linear_model.summary(ci=True, alpha=0.05)
    
    assert isinstance(summary_df, pd.DataFrame), "Summary output should be a DataFrame."
    assert all(
        col in summary_df.columns
        for col in ["Parameter", "Estimate", "Std. Error", "T-Statistic", "P-Value", "CI Lower", "CI Upper"]
    ), "Summary should include confidence intervals when requested."
    
def test_summary_output_without_ci(example_data):
    """
    Test if summary outputs the correct DataFrame structure without confidence intervals.
    """
    X, y = example_data
    model = LinearModel()
    model.fit(X, y)
    model.get_std_error()
    model.get_test_statistic()
    summary = model.summary(ci=False)
    
    assert isinstance(summary, pd.DataFrame), "Summary should return a DataFrame."
    
    assert "Parameter" in summary.columns, "Summary should include 'Parameter' column."
    assert "Estimate" in summary.columns, "Summary should include 'Estimate' column."
    assert "Std. Error" in summary.columns, "Summary should include 'Std. Error' column."
    assert "T-Statistic" in summary.columns, "Summary should include 't-value' column."
    assert "P-Value" in summary.columns, "Summary should include 'p-value' column."
    assert "CI Lower" not in summary.columns, "Confidence intervals should not be included when ci=False."
    assert "CI Upper" not in summary.columns, "Confidence intervals should not be included when ci=False."



def test_summary_invalid_alpha(linear_model, example_data):
    """Test that an error is raised for invalid alpha values."""
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
    """Test that the summary values are correct."""
    X, y = example_data
    linear_model.fit(X, y)
    summary_df = linear_model.summary(ci=True, alpha=0.05)
    
    # Use statsmodels to calculate expected values
   
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()
    
    # Replace expected values with those from statsmodels
    expected_params = model.params.values  # Intercept and slope
    expected_pvalues = model.pvalues  # P-values for parameters
    
    # Check parameter estimates
    assert np.allclose(summary_df["Estimate"].values, expected_params, atol=0.1), \
        f"Parameter estimates are incorrect: {summary_df['Estimate'].values}."
  