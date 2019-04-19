# -*- coding: utf-8 -*-

# MF 796 - Assignment 5 Problem 1
# Lingyi Xu, U77017242
# Date: 2019-03-03



# Problem 1
# (a)
import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve

def calculate_d1(S0, K, r, t, sig):
    nu = np.log(S0/K)+(r+0.5*sig**2)*t
    de = sig*np.sqrt(t)
    return nu/de

S0 = 100.0
r = 0.0
t1 = 1.0/12
t2 = 3.0/12
one_mon = {-10:0.3325, -25:0.2473, -40:0.2021, 40:0.1571, 25:0.1370, 10:0.1148}
thr_mon = {-10:0.2836, -25:0.2178, -40:0.1818, 40:0.1462, 25:0.1256, 10:0.1094}

# calculate delta in terms of strike
klist1 = []
klist2 = []
for key,sig in one_mon.items():
    def myfunc1(K):
        return norm.cdf(calculate_d1(S0, K, r, t1, sig))-int(abs(key)!=key)-key/100.0
    klist1.append(fsolve(myfunc1, S0))
for key,sig in thr_mon.items():
    def myfunc2(K):
        return norm.cdf(calculate_d1(S0, K, r, t2, sig))-int(abs(key)!=key)-key/100.0
    klist2.append(fsolve(myfunc2, S0))
  

# (b)
import matplotlib.pyplot as plt

strike_1mon = []
strike_3mon = []
for i in range(len(klist1)):
    strike_1mon += list(klist1[i])
    strike_3mon += list(klist2[i])

vol_1mon = list(one_mon.values())
vol_3mon = list(thr_mon.values())
curve1 = np.polyfit(strike_1mon, vol_1mon, 3)
curve2 = np.polyfit(strike_3mon, vol_3mon, 3)

x = np.arange(84, 108, 0.01)
y1 = np.polyval(curve1, x)
y2 = np.polyval(curve2, x)
plt.plot(x, y1, label="1 month")
plt.plot(x, y2, label="3 month")
plt.title('volatility function for all strikes')
plt.xlabel('strike')
plt.ylabel('volatility')
plt.legend()
plt.show()


"""
from scipy.interpolate import CubicSpline as cs
curve1 = cs(strike_1mon, vol_1mon)
curve2 = cs(strike_3mon, vol_3mon)
xs = np.arange(84,108,0.01)
plt.plot(xs, curve1(xs), label='1 month')
plt.plot(xs, curve2(xs), label='3 month')
"""


# (c)
def call_price(K, sig):
    d1 = 1.0/(sig*np.sqrt(t1))*(np.log(S0/K)+(r+.5*(sig**2))*t1)
    d2 = d1 - sig*np.sqrt(t1)
    price = S0*norm.cdf(d1) - np.exp(-r*t1)*K*norm.cdf(d2)
    return price

def rn_prob1(K, delta=0.01):
    sig = np.polyval(curve1, K)
    sigl = np.polyval(curve1, K-delta)
    sigu = np.polyval(curve1, K+delta)
    return (call_price(K-delta, sigl)+call_price(K+delta, sigu)-2*call_price(K, sig))/delta**2

def rn_prob2(K, delta=0.01):
    sig = np.polyval(curve2, K)
    sigl = np.polyval(curve2, K-delta)
    sigu = np.polyval(curve2, K+delta)
    return (call_price(K-delta, sigl)+call_price(K+delta, sigu)-2*call_price(K, sig))/delta**2

x = np.arange(80, 125, 0.01)
xs = x[1:-1]
plt.plot(xs, rn_prob1(xs), label='1 month')
plt.plot(xs, rn_prob2(xs), label='3 month')
plt.title('risk neutral density for 1&3 month options')
plt.xlabel('terminal price')
plt.ylabel('density')
plt.legend()
plt.plot()


# (d)
const_sig1 = 0.1824
const_sig2 = 0.1645
def rn_constvol_prob1(K, delta=0.01):
    return (call_price(K-delta, const_sig1)+call_price(K+delta, const_sig1)-2*call_price(K, const_sig1))/delta**2
def rn_constvol_prob2(K, delta=0.01):
    return (call_price(K-delta, const_sig2)+call_price(K+delta, const_sig2)-2*call_price(K, const_sig2))/delta**2

plt.plot(xs, rn_constvol_prob1(xs), label='1 month')
plt.plot(xs, rn_constvol_prob2(xs), label='3 month')
plt.title('risk neutral density with constant vol for 1&3 month options')
plt.xlabel('terminal price')
plt.ylabel('density')
plt.legend()
plt.plot()

# put four distributions in a same graph
plt.plot(xs, rn_prob1(xs), label='1 month')
plt.plot(xs, rn_prob2(xs), label='3 month')
plt.plot(xs, rn_constvol_prob1(xs), label='1 month with const vol')
plt.plot(xs, rn_constvol_prob2(xs), label='3 month with const vol')
plt.title('risk neutral density for 1&3 month options w/o constant vol')
plt.xlabel('terminal price')
plt.ylabel('density')
plt.legend()
plt.plot()


# (e)
price_1mondp = 0
price_3mondc = 0
price_2monc = 0
for st in np.arange(80, 110, 0.01):
    price_1mondp += rn_prob1(st)*0.01
for st in np.arange(105, 125, 0.01):
    price_3mondc += rn_prob2(st)*0.01
for st in np.arange(100, 125, 0.01):
    price_2monc += (rn_prob1(st)+rn_prob2(st))/2*(st-100)*0.01

  