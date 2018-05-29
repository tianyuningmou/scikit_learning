# -*- coding: utf-8 -*-

"""
Copyright () 2018

All rights reserved

FILE: LARS_Lasso.py
AUTHOR: tianyuningmou
DATE CREATED:  @Time : 2018/5/29 下午4:11

DESCRIPTION:  .

VERSION: : #1 
CHANGED By: : tianyuningmou
CHANGE:  : 
MODIFIED: : @Time : 2018/5/29 下午4:11
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn import datasets


diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

print("Computing regularization path using the LARS ...")
alphas, _, coefs = linear_model.lars_path(X, y, method='lasso', verbose=True)

xx = np.sum(np.abs(coefs.T), axis=1)
xx /= xx[-1]

plt.plot(xx, coefs.T)
ymin, ymax = plt.ylim()
plt.vlines(xx, ymin, ymax, linestyle='dashed')
plt.xlabel('|coef| / max|coef|')
plt.ylabel('Coefficients')
plt.title('LASSO Path')
plt.axis('tight')
plt.show()
