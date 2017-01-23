# ANFIS with correlated residuals

ANFIS refers to 'adaptive network-based fuzzy inference system' which fits a Sugeno-type fuzzy model to a training dataset. 

This Python project ```anfis_co``` deals with the situation when there is a certain degree of **correlation** between the residuals in an **'Output = ANFIS(Input) + Residuals'** model. For example, under many conditions, the residuals may have spatial or temporal correlation. 

The regular anfis package in Python can be found at [https://github.com/twmeggs/anfis](https://github.com/twmeggs/anfis), from whch this project was modified. 

## Installation


Requires Python >= 3.0. (All the modules were tested in Python 3.5.2.)

This project was not published as a package in PyPI and cannot be installed by running ```$ pip install```. Therefore, in order to use the package, you need to download the files and use them as a self-written package. 

`anfis_co` depends on: 

* numpy
* sklearn
* scikit-fuzzy
* matplotlib

## Usage
* `anfis_co.py`

  Input: {**x**: independent variables in training data, **y**: response in training data, **C**: covariance matrix of residuals at different locations or time}.
  
  Output: parameters of fuzzy inference system fitted to training data **{x, y, C}**.

* `test_spatial.py`

  In most conditions, covariance matrix **C** is unknown and need to be estimated. Therefore, an iterative method need to be used. ```test_spatial``` gives an example showing how to conduct an iterative algorithm for (a) spatial kernal estimation using ```sklearn.gaussian_process```, and (b) fuzzy inference system estimation using ```anfis_co```. 

## Notes

 * Check [generalized least squares](https://en.wikipedia.org/wiki/Generalized_least_squares) in Wikipedia to see how the correlation in residuals affecting the inference of model parameters.  
 * Make **C = I** if no correlation exists in residuals. In this case, ```anfis_co``` will produce the same results with ```anfis``` in original anfis package at [https://github.com/twmeggs/anfis](https://github.com/twmeggs/anfis). 
