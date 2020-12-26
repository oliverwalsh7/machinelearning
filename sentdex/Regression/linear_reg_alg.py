# Linear Regression Algorithm
# really only useful for skewed data sets

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

def create_dataset(hm, variance, step = 2, correlation = False):
    val = 1
    ys = []

    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype = np.float64), np.array(ys, dtype = np.float64)


def best_fit(xs, ys): # slope and intercept, m & b
    m = (((mean(xs) * mean(ys)) - mean(xs * ys)) /
          ((mean(xs)**2) - mean(xs**2))) # slope of best fi line
    b = mean(ys) - m*mean(xs)
    return m, b

def squared_error(orig_ys, ys_line):
    return sum((ys_line - orig_ys)**2)

def coefficient_of_determination(orig_ys, ys_line):
    y_mean_line = [mean(orig_ys) for y in orig_ys]
    squared_err_reg = squared_error(orig_ys, ys_line)
    squared_err_y_mean = squared_error(orig_ys, y_mean_line)
    return 1 - (squared_err_reg / squared_err_y_mean)

xs, ys = create_dataset(40, 40, 2, correlation='pos')

m, b = best_fit(xs, ys)
regression_line = [(m*x) + b for x in xs]
r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

plt.scatter(xs, ys)
#plt.scatter(predict_x, predict_y, color = "g")
plt.plot(xs, regression_line)
plt.show()