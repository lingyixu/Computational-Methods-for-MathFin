

#######################################################################
# Three quadrature rules: Left Riemann rule, Midpoint rule, Gauss nodes
# Compare quadrature results with BS results
#######################################################################


# 1. Pricing under BS formula
import numpy as np
import scipy.stats as sc
import numpy.polynomial.legendre as pl

def bs_price(S0,K,t,r,sigma):
    # calculate an option price via Black-Scholes Formula
    
    d1 = 1.0/(sigma*np.sqrt(t))*(np.log(S0/K)+(r+.5*(sigma**2))*t)
    d2 = d1 - sigma*np.sqrt(t)
    BS_price = S0*sc.norm.cdf(d1) - np.exp(-r*t)*K*sc.norm.cdf(d2)
    
    return BS_price

S0 = 10
K = 12
t = 3/12.0
r = .04
sigma = .2

call_price = bs_price(S0,K,t,r,sigma)



# 2. Pricing using quadrature methods
# dSt = miu * St * dt + sigma * St * dWt, Wt~N(0, t)
# St = S0 * exp((r - 0.5 sigma^2)t + sigma * Wt)
# ln(St) ~ N (ln(S0) + (r - 0.5 sigma^2)t, sigma^2 * t)
# integral w.r.t. ln(St)

# calculate the mean and std of ln(St)
i_miu = np.log(S0) + (r-0.5*sigma**2)*t
i_std = np.sqrt(t)*sigma

# set intergral boundaries
a = np.log(K)
b = i_miu + 5*i_std

def left_rule(node_num, lower, upper):
    interval = (upper-lower)/node_num
    return (lower + interval*np.arange(node_num))

def mid_rule(node_num, lower, upper):
    interval = (upper-lower)/node_num
    return (lower + interval*0.5 + np.arange(node_num)*interval)

node_list = [5,10,50,100]


######## left Riemann rule #######
l_val = {}
for node in node_list:
    left_node = left_rule(node, a, b)
    l_sum = 0
    for l_node in left_node:
        l_sum += np.exp(-r*t)*(np.exp(l_node)-K)*sc.norm.pdf(l_node, i_miu, i_std)*(b-a)/node
    l_val[node] = [l_sum, call_price-l_sum]

    
######## midpoint rule ########
m_val = {}
for node in node_list:
    mid_node = mid_rule(node, a, b)
    m_sum = 0
    for m_node in mid_node:
        m_sum += np.exp(-r*t)*(np.exp(m_node)-K)*sc.norm.pdf(m_node, i_miu, i_std)*(b-a)/node
    m_val[node] = [m_sum, call_price-m_sum]

    
######## Gauss nodes ########
g_val = {}
for node in node_list:
    nodes, weights = pl.leggauss(node)
    nodes = nodes*(b-a)/2 + (b+a)/2
    g_sum = 0
    for i in np.arange(node):
        g_sum += np.exp(-r*t)*(np.exp(nodes[i])-K)*sc.norm.pdf(nodes[i], i_miu, i_std)*(b-a)/2*weights[i]
    g_val[node] = [g_sum, call_price-g_sum]

