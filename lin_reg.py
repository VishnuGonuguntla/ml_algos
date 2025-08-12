import numpy as np
# import scipy.linalg as scipy
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from include import LinearRegression

n = 100

x = 100*np.random.rand(n)
y = 100*np.random.rand(n)

x = np.sort(x)
y = np.sort(y)
data = {}

data["x"] = x
data["y"] = y

# Hyper Parameters:
learning_rate = 0.0001
n = 100

reg = LinearRegression.LinearReg(learning_rate, n, data)