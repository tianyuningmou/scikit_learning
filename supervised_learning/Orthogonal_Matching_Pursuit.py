# -*- coding: utf-8 -*-

"""
Copyright () 2018

All rights reserved

FILE: Orthogonal_Matching_Pursuit.py
AUTHOR: tianyuningmou
DATE CREATED:  @Time : 2018/5/29 下午4:42

DESCRIPTION:  .

VERSION: : #1 
CHANGED By: : tianyuningmou
CHANGE:  : 
MODIFIED: : @Time : 2018/5/29 下午4:42
"""

import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.datasets import make_sparse_coded_signal


n_components, n_features = 512, 100
n_nonzero_coefs = 17
y, X, w = make_sparse_coded_signal(n_samples=1, n_components=n_components, n_features=n_features,
                                   n_nonzero_coefs=n_nonzero_coefs, random_state=0)
idx, = w.nonzero()
y_noisy = y + 0.05 * np.random.randn(len(y))
plt.figure(figsize=(7, 7))
plt.subplot(4, 1, 1)
plt.xlim(0, 512)
plt.title('Sparse signals')
plt.stem(idx, w[idx])

omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
omp.fit(X, y)
coef = omp.coef_
idx_r, = coef.nonzero()
plt.subplot(4, 1, 2)
plt.xlim(0, 512)
plt.title("Recovered signal from noise-free measurements")
plt.stem(idx_r, coef[idx_r])

omp.fit(X, y_noisy)
coef = omp.coef_
idx_r, = coef.nonzero()
plt.subplot(4, 1, 3)
plt.xlim(0, 512)
plt.title("Recovered signal from noisy measurements")
plt.stem(idx_r, coef[idx_r])

omp_cv = OrthogonalMatchingPursuitCV()
omp_cv.fit(X, y_noisy)
coef = omp_cv.coef_
idx_r, = coef.nonzero()
plt.subplot(4, 1, 4)
plt.xlim(0, 512)
plt.title("Recovered signal from noisy measurements with CV")
plt.stem(idx_r, coef[idx_r])
plt.subplots_adjust(0.06, 0.04, 0.94, 0.90, 0.20, 0.38)
plt.suptitle('Sparse signal recovery with Orthogonal Matching Pursuit', fontsize=16)
plt.show()
