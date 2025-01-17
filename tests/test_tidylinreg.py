from tidylinreg.tidylinreg import LinearModel
from tidylinreg.tidylinreg import LinearModel
import pandas as pd
import numpy as np
from scipy.stats import norm
import pytest


# DO NOT CHANGE SEED: some tests will fail
# DO NOT CHANGE SEED: some tests will fail
SEED = 524

# creating test data

# creating test data

## Simple Linear Regression ##
X_slr = pd.DataFrame({'x':[-2,-1,0,1,2]})

# perfect line with no error: y = 3x + 2
y_perfect_line = 3*X_slr.squeeze() + 2

# perfect line with no intercept: y = -4x
y_no_intercept = -4*X_slr.squeeze()

# perfect line with zero slope (a constant) y = 4
y_const = pd.Series([4,4,4,4,4])

# a line y = 3x + 2 with normally distributed errors
y_noisy_line = 3*X_slr.squeeze() + 2 + norm.rvs(random_state=SEED)


## Multiple Linear Regression ##
X_mlr = pd.DataFrame({
    'x1':[0,8,3,5],
    'x2':[2,7,1,8],
    'x3':[8,1,3,5]
})

# a perfect linear function: y = 3x1 + 4x2 + 5x3 + 6
y_mlr_perfect = 3*X_mlr['x1'] + 4*X_mlr['x2'] + 5*X_mlr['x3'] + 6

# a linear function where one slope is zero: y = 3x1 + 4x2 + 0x3 + 6
y_one_zero_slope = 3*X_mlr['x1'] + 4*X_mlr['x2'] + 6

# a constant y = 6
y_mlr_const = pd.Series(6*np.ones(4))
# a constant y = 6
y_mlr_const = pd.Series(6*np.ones(4))

# a linear function y = 3x1 + 4x2 + 5x3 + 6, with noise
y_mlr_noise = 3*X_mlr['x1'] + 4*X_mlr['x2'] + 5*X_mlr['x3'] + 6 + norm.rvs(size=4,random_state=SEED)


## Edge Cases and Adversarial Usage ##
# a linear function y = 3x1 + 4x2 + 5x3 + 6, with noise
y_mlr_noise = 3*X_mlr['x1'] + 4*X_mlr['x2'] + 5*X_mlr['x3'] + 6 + norm.rvs(size=4,random_state=SEED)


## Edge Cases and Adversarial Usage ##

# a design matrix with categorical variables
X_mlr_categorical = pd.DataFrame({
    'x1':[0,8,3,5],
    'x2':[2,7,1,8],
    'x3':[8,1,3,5],
    'x4':['blue','red','red','blue']
})

# a categorical response:
# a categorical response:
y_categorical = pd.Series(['red','blue','blue','yellow','blue'])

# a singular design matrix (should throw LinAlgError when OLS estimate calculated)
X_collinear = pd.DataFrame({
    'x1':[1,1,1,1],
    'x2':[2,2,2,2],
    'x3':[0,1,2,3]
})

# Response that doesn't match shape of X_mlr
y_mlr_wrong_size = pd.Series([1,2,3,4,5,6,7,8,9,10])

# Design Matrix with a missing entry
X_has_nan = pd.DataFrame({
    'x1':[0,np.nan,3,5],
    'x2':[2,7,1,8],
    'x3':[8,1,3,5]
})

# Response with a missing entry
y_has_nan = pd.Series([7,8,np.nan,10])

# Response and explanatory variables with only one sample
y_one_sample = pd.Series([1])
X_slr_one_sample = pd.DataFrame({'x':[-2]})
X_mlr_one_sample = pd.DataFrame({
    'x1':[0],
    'x2':[2],
    'x3':[8]
})

# Response and explanatory variable with no samples
y_empty = pd.Series([])
X_slr_empty = pd.DataFrame({'x':[]})
X_mlr_one_sample = pd.DataFrame({
    'x1':[],
    'x2':[],
    'x3':[],
})

# initalize a linear model
@pytest.fixture
def linear_model():
    return LinearModel()

# test fitting on linear regression gives expected parameter estimates
@pytest.mark.parametrize(
    'X,y,expected_params',
    list(zip([X_slr for _ in range(4)] + [X_mlr for _ in range(4)],
    [y_perfect_line,y_no_intercept,y_const,y_noisy_line,
     y_mlr_perfect,y_one_zero_slope,y_mlr_const,y_mlr_noise],
    [pd.Series([2,3],index=['(Intercept)','x']),
     pd.Series([0,-4],index=['(Intercept)','x']),
     pd.Series([4,0],index=['(Intercept)','x']),
     pd.Series([0.568,3],index=['(Intercept)','x']),
     pd.Series([6,3,4,5],index=['(Intercept)','x1','x2','x3']),
     pd.Series([6,3,4,0],index=['(Intercept)','x1','x2','x3']),
     pd.Series([6,0,0,0],index=['(Intercept)','x1','x2','x3']),
     pd.Series([28.978,-1.172, 6.227, 1.392],index=['(Intercept)','x1','x2','x3'])]))
)
def test_fit_params(linear_model,X,y,expected_params):
    linear_model.fit(X,y)
    assert np.allclose(linear_model.params,expected_params,atol=0.001)
    assert (linear_model.param_names == expected_params.index).all()

# test that error is correctly thrown for incorrect use cases of fit
@pytest.mark.parametrize(
    'X,y,expected_error',
    [(X_mlr_categorical,y_mlr_perfect,TypeError),     # categorical features in X
     (X_mlr,y_categorical,TypeError),                 # categorical y
     (X_collinear,y_mlr_perfect,ValueError),          # collinear X
     (X_has_nan,y_mlr_perfect,ValueError),            # missing entries in X
     (X_mlr,y_has_nan,ValueError),                     # missing entries in y
     (X_mlr,y_mlr_wrong_size,ValueError),             # mismatching sizes of X and y
     (X_mlr_one_sample,y_mlr_perfect,ValueError),     # X only has one sample
     (X_slr_one_sample,y_perfect_line,ValueError),    # X has one sample and one feature
     (X_mlr,y_one_sample,ValueError),                 # y only has one sample
     (X_mlr,y_empty,ValueError),                      # y has no samples
     (X_slr_empty,y_perfect_line,ValueError),         # X has no samples
     ]
)
def test_fit_throw_error(linear_model,X,y,expected_error):
    with pytest.raises(expected_error):
        linear_model.fit(X,y)

# creating prediction test data
X_slr_test = pd.DataFrame({'x':[3,4,5,6,7]})
y_perfect_line_test = 3*X_slr_test.squeeze() + 2

X_mlr_test = pd.DataFrame({
    'x1':[1,8,4,6],
    'x2':[3,8,2,9],
    'x3':[4,2,4,6]
})
y_mlr_perfect_test = 3*X_mlr_test['x1'] + 4*X_mlr_test['x2'] + 5*X_mlr_test['x3'] + 6

# test for accurate model predictions
@pytest.mark.parametrize(
    'X_fit,y_fit,X_test,expected_predictions',
    [(X_slr,y_perfect_line,X_slr,y_perfect_line),               # in-sample predictions (SLR)
     (X_mlr,y_mlr_perfect,X_mlr,y_mlr_perfect),                 # in-sample predictions (MLR)
     (X_slr,y_perfect_line,X_slr_test,y_perfect_line_test),     # out-of-sample predictions (SLR)
     (X_mlr,y_mlr_perfect,X_mlr_test,y_mlr_perfect_test)        # out-of-sample predictions (MLR)
    ]
)
def test_predict(linear_model,X_fit,y_fit,X_test,expected_predictions):
    linear_model.fit(X_fit,y_fit)
    predictions = linear_model.predict(X_test)
    assert np.allclose(predictions,expected_predictions,atol=0.001)

# test that error is thrown when prediction attempted without fitting
def test_predict_throw_error():
    linear_model = LinearModel()
    with pytest.raises(ValueError):
        linear_model.predict(X_slr)