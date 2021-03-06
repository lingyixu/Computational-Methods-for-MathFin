
####################### BLACK-SCHOLES PDE #######################
# ∂c/∂t + 0.5*sig^2*s^2*(∂^2c/∂t^2) + r*s*∂c/∂s - r*c = 0
#################################################################


import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


########################## PART 1 ##########################
# estimate a sgima without using calibrtaion
############################################################

# find sigma from option price

def bs_vol(S0,K,t,r,p):
    
    def price(sig):
        d1 = 1.0/(sig*np.sqrt(t))*(np.log(S0/K)+(r+.5*(sig**2))*t)
        d2 = d1 - sig*np.sqrt(t)
        call = S0*norm.cdf(d1) - np.exp(-r*t)*K*norm.cdf(d2)
        return call-p
    
    return fsolve(price, 0.2)[0]


# Plug in real market values and use arithmetic average

sig1 = bs_vol(277.33, 285, 208.0/365, .0247, 9.59)
sig2 = bs_vol(277.33, 290, 208.0/365, .0247, 6.88)
sig = (sig1 + sig2)/2




########################## PART 2 ##########################
# explicit Euler discretization of BS pde
############################################################

# define a function to recover call prices, and get matrix A, eigenvalues as well as relative error

def sol_eq(s0,smax,k,T,N,M,sig,r,p,flag='European'):
    
    # spot stock price: S0
    # upper boundary chosen for stock price: smax
    # strike price: k
    # time horizon: T
    # number of time grids: N
    # number of strike grids: M
    # stock volatility (constant): sig
    # interest rate: r
    # spot option price: p
    # flag: option type (European/American)
    
    # construct grids for time and strike
    ht = T/N
    hs = smax/M
    t = np.arange(0, T+ht, ht)
    s = np.arange(0, smax+hs, hs)
    
    # assign initial values in matrix A
    a = 1-(sig**2)*(s**2)*ht/(hs**2)-r*ht
    l = 0.5*(sig**2)*(s**2)*ht/(hs**2)-r*s*ht/(2*hs)
    u = 0.5*(sig**2)*(s**2)*ht/(hs**2)+r*s*ht/(2*hs)
    
    # construct matrix A
    A = np.matrix(np.zeros((M-1,M-1)))
    diag = a[1:]
    upperDiag = u[1:M-1]
    lowerDiag = l[2:M]
    for i in range(len(upperDiag)):
        A[i,i+1] = upperDiag[i]
        A[i+1,i] = lowerDiag[i]
    for i in range(M-1):
        A[i,i] = diag[i]
    vec_eigenvalue = np.linalg.eigvals(A)
    
    # calculate vector b
    b = u[M-1]*(s[M]-k*np.exp(-r*(T-t)))
    ba = u[M-1]*(s[M]-k)
    
    diff = s-k
    diff[diff<0]=0
    ter_c = np.matrix(diff[1:M]).T
    
    # calcualte price for European and American calls respectively
    if flag == 'European':
        pile = np.matrix(np.zeros(M-1)).T
        for i in range(N):
            bb = np.matrix(np.append(np.zeros(M-2),b[i+1])).T
            pile += (A**i)*bb
        vec_c = pile + (A**N)*ter_c
    
    if flag == 'American':
        pile = np.matrix(np.zeros(M-1)).T
        for i in range(N):
            bb = np.matrix(np.append(np.zeros(M-2),ba)).T
            pile += (A**i)*bb
        vec_c = pile + (A**N)*ter_c
    
    # calculate relative error
    c = np.squeeze(np.asarray(vec_c))
    index = int(M/(smax/s0)-1)
    error = abs(c[index]-p)/p
    
    return A, abs(vec_eigenvalue), vec_c, error


# plug in real market values
s0,k1,k2,T,sig,r,p = 277.33,285,290,208.0/365,(sig1+sig2)/2,0.0247,9.59


# change N to see convergence
smax,M = 277.33*5,500
err1 = {}
for N in [500, 1000, 2000, 3000, 5000, 10000]:
    _, _, _, err1[N] = sol_eq(s0,smax,k1,T,N,M,sig,r,p)

    
# change M to see convergence
smax,N = 277.33*5,5000
err2 = {}
for M in [50, 100, 200, 500, 700, 1000]:
    _, _, _, err2[M] = sol_eq(s0,smax,k1,T,N,M,sig,r,p)

    
# change smax to see convergence
N,M = 5000,500
err3 = {}
for smax in [s0,2*s0,5*s0,10*s0]:
    _, _, _, err3[smax] = sol_eq(s0,smax,k1,T,N,M,sig,r,p)


    
    
########################## PART 3 ##########################
# check stability of matrix A: plot eigenvalues
############################################################    

smax,N,M = s0*5,5000,500
A, v1, _, _ = sol_eq(s0,smax,k1,T,N,M,sig,r,p)
plt.plot(sorted(v1,reverse=True))
plt.title('matrix A eigenvalue')




########################## PART 4 ##########################
# calcualte call spread with strike k1 and k2
############################################################

#  for European calls:

_, _, c1, _ = sol_eq(s0,smax,k1,T,N,M,sig,r,p)
_, _, c2, _ = sol_eq(s0,smax,k2,T,N,M,sig,r,p)
arr1 = np.squeeze(np.asarray(c1))
arr2 = np.squeeze(np.asarray(c2))
spread1 = abs(arr1[int(M/5-1)]-arr2[int(M/5-1)])    # by construction, the spot option price lies in the (M/5)th position


# for American calls:

_, _, ca1, _ = sol_eq(s0,smax,k1,T,N,M,sig,r,p,'American')
_, _, ca2, _ = sol_eq(s0,smax,k2,T,N,M,sig,r,p,'American')
a1 = np.squeeze(np.asarray(ca1))
a2 = np.squeeze(np.asarray(ca2))
spread2 = abs(a1[int(M/5-1)]-a2[int(M/5-1)])

