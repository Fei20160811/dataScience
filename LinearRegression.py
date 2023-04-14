#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 15:24:47 2023

@author: ml
"""
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression

rng = np.random.RandomState(1)
#numpy.random.RandomState.rand 
#Create an array of the given shape and populate it 
#   with random samples from a uniform distribution over [0, 1)
#numpy.random.RandomState.rand 產生50組均勻分佈的隨機數
x = 10 * rng.rand(50)
y = 2 * x - 5 + rng.randn(50)

plt.scatter(x, y)


model = LinearRegression(fit_intercept=True)

model.fit(x[:, np.newaxis], y)

xfit = np.linspace(0, 10, 100)
yfit = model.predict(xfit[:, np.newaxis])

plt.scatter(x, y)
plt.plot(xfit, yfit)

print("Model slope:", model.coef_[0])
print("Model intercept", model.intercept_)

