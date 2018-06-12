# -*- coding: utf-8 -*-

"""
Copyright () 2018

All rights reserved

FILE: LinearRegression.py
AUTHOR: tianyuningmou
DATE CREATED:  @Time : 2018/5/25 下午3:00

DESCRIPTION:  .

VERSION: : #1 
CHANGED By: : tianyuningmou
CHANGE:  : 
MODIFIED: : @Time : 2018/5/25 下午3:00
"""

"""
最小二乘法的复杂度：如果X是一个size为(n, p)的矩阵，设n>=p，则该方法的复杂度为O(np^2)
"""

from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


diabetes = datasets.load_diabetes()
diabetes_X = diabetes.data[:, np.newaxis, 2]
diabetes_X_train = diabetes_X[: -20]
diabetes_X_test = diabetes_X[-20:]
diabetes_Y_train = diabetes.target[: -20]
diabetes_Y_test = diabetes.target[-20:]

# diabetes_X_train = [[1], [2], [3]]
# diabetes_X_test = [[4], [5]]
# diabetes_Y_train = [[1], [2], [3]]
# diabetes_Y_test = [[4], [5]]

model = linear_model.LinearRegression()
model.fit(diabetes_X_train, diabetes_Y_train)
diabetes_Y_pred = model.predict(diabetes_X_test)

print('Coefficients: {C}'.format(C=model.coef_))
print('Mean squared error: {M}'.format(M=mean_squared_error(diabetes_Y_test, diabetes_Y_pred)))
print('Variance score: {V}'.format(V=r2_score(diabetes_Y_test, diabetes_Y_pred)))

plt.scatter(diabetes_X_test, diabetes_Y_test, color='black')
plt.plot(diabetes_X_test, diabetes_Y_pred, color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()
