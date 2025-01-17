from tidylinreg.tidylinreg import LinearModel
import pandas as pd
import numpy as np
from scipy.stats import norm

SEED = 524
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

# a linear function y = 3x1 + 4x2 + 5x3 + 6 with with normally distributed errors
y_mlr_noisy = 3*X_mlr['x1'] + 4*X_mlr['x2'] + 5*X_mlr['x3'] + 6 + norm.rvs(size=3,random_state=SEED)

# a linear function where one parameter is zero: y = 3x1 + 0x2 + 5x3 + 6, with noise
y_mlr_zero_slope_noise = 3*X_mlr['x1']+ 5*X_mlr['x3'] + 6 + norm.rvs(size=3,random_state=SEED)

# a design matrix with categorical variables
X_mlr = pd.DataFrame({
    'x1':[0,8,3,5],
    'x2':[2,7,1,8],
    'x3':[8,1,3,5],
    'x4':['blue','red','red','blue']
})


## Edge Cases and Adversarial Usage ##

# categorical response:
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
    'x3':[]
})






model = LinearModel()
model.params = [2.0]
model.std_error = [0.5]
def test_get_test_statistic(model):
    # test basic functionality
    expected = model.params / model.std_error
    assert model.get_test_statistic() == pytest.approx(expected_t_stat, rel=1e-6)

    # test perfect fit
    pass



def test_get_test_statistic_error():
    pass



def test_get_pvalues():
    # Error: invalid with less than 8 observations
    # Warning: p-value may be inaccurate with fewer than x (=20) observations; only X observations given
    
    ## SLR
    #   test perfect line with no error: y = 3x + 2
    #   - p-val should be 0 for all estimates
    #   test perfect line with no intercept: y = -4x
    #   - p-val should be 0 for all estimates
    #   test perfect line with zero slope (a constant) y = 4
    #   - TODO: find expected behaviour
    #   test a line y = 3x + 2 with normally distributed errors
    #   - TODO: find expected behaviour

    ## Edge cases and adverserial usage
    ## ** Probably don't need to include since these will all be
    #       covered by fit and predict, which is required for p-vals **
    #   test categorical response
    #   - throws TypeError
    #   test response that doesn't match shape of X_mlr
    #   - throws Error (TBD)
    #   test x with a missing entry
    #   - throws ValueError
    #   test response with a missing entry
    #   - produces Warning but not error
    #   test response and explanatory variables with only one sample
    #   - throws ValueError
    #   test response and explanatory variable with no samples
    #   - throws ValueError
    pass



def test_get_pvalues_error():
    pass