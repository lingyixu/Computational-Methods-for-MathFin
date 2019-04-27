
###################### HESTON MODEL ########################
# dSt = (r-q)*St*dt + vt^0.5*St*dWt1              --- (1)
# dvt = k*(theta-vt) + sig*vt^0.5*dWt2            --- (2)
# Cov(dWt1, dWt2) = rho*dt                        --- (3)
############################################################


import numpy as np
import matplotlib.pyplot as plt


######################### PART 1 ###########################
# European option simulation under Heston model 
############################################################


#########################################
############# FUNCTION ZONE #############
#########################################

def Euro_simu(S0,K,q,r,T,v0,sig,k,theta,rho,n,N):
    
    # spot price: S0
    # strike price: K
    # dividend rate: q
    # interest rate: r
    # initial variance: v0
    # stochastic term coefficient: sig
    # volatility mean reverting speed: k
    # long term volatility level: theta
    # stochastic term correlation: rho
    # number of simulations: n
    # number of time intervals: N
    
    dt = T/N
    mean = (0,0)
    cov = [[dt,rho*dt],[rho*dt,dt]]
    st_vec = np.empty(n)
    vt_vec = np.empty(n)
    payoff_vec = np.empty(n)
    price_vec = np.empty(n)

    for i in range(n):
    
        x = np.random.multivariate_normal(mean,cov,(N,1))
        st = S0
        vt = v0
    
        for j in range(N):
            dst = (r-q)*st*dt + np.sqrt(vt)*st*x[j][0][0]
            dvt = k*(theta-vt)*dt + sig*np.sqrt(vt)*x[j][0][1]
            st += dst
            vt += dvt
            if vt<0:
                vt = -vt    # implicit boundary: vt>=0
    
        payoff_vec[i] = max(st-K1,0)
        price_vec[i] = payoff_vec[i]*np.exp(-r*T)
        st_vec[i] = st
        vt_vec[i] = vt
        
    plt.hist(st_vec,bins=100)
    plt.title("Terminal stock price distribution")
    plt.xlabel("stock price")
    plt.ylabel("frequncy per " + str(n) + " trials")
    plt.show()

    return price_vec


##########################################
############### INPUT ZONE ###############
##########################################

# The following parameter values come from calibration of real market data

S0 = 282
K1 = 285
K2 = 315
q = 1.77e-02
r = 1.5e-02
T = 1
v0 = 0.05
sig = 0.45
k = 0.04
theta = 0.19
rho = 0

n = 1000
N = 5000

price1 = Euro_simu(S0,K1,q,r,T,v0,sig,k,theta,rho,n,N)
miu1 = np.mean(price1)
sig1 = np.std(price1)


##########################################
####### RELATIVE ERROR CALCULATION #######
##########################################

n1 = np.mean(price1[:-1])
n2 = np.mean(price1)
err_pct = abs(n2-n1)/n1*100


err = np.empty(0)

for N in [1000, 2000, 5000, 10000, 20000]:
    
    # pay attention to relative error convergence as N grows larger
    
    price = Euro_simu(S0,K1,q,r,T,v0,sig,k,theta,rho,n,N)
    miu = np.mean(price)
    sig = np.std(price)
    
    n1 = np.mean(price[:-1])
    n2 = np.mean(price)
    err_pct = abs(n2-n1)/n1*100
    err = np.append(err, err_pct)



    
    
######################### PART 2 ###########################
# Knock-out call simulation under Heston model 
############################################################ 


#########################################
############# FUNCTION ZONE #############
#########################################


def knock_simu(S0,K1,K2,q,r,T,v0,sig,k,theta,rho,n,N):
    
    # K1: strike price
    # K2: upper bound for underlying stock price

    dt = T/N
    mean = (0,0)
    cov = [[dt,rho*dt],[rho*dt,dt]]
    st_vec = np.empty(n)
    vt_vec = np.empty(n)
    payoff_vec = np.empty(n)
    price_vec = np.empty(n)

    for i in range(n):

        x = np.random.multivariate_normal(mean,cov,(N,1))
        st = S0
        vt = v0
        flag = 1

        for j in range(N):
            dst = (r-q)*st*dt + np.sqrt(vt)*st*x[j][0][0]
            dvt = k*(theta-vt)*dt + sig*np.sqrt(vt)*x[j][0][1]
            st += dst
            vt += dvt
            if vt<0:
                vt = -vt
            if st>=K2:
                flag = 0
    
        if flag==0:
            payoff = 0
        else:
            payoff = max(st-K1,0)
        payoff_vec[i] = payoff
        price_vec[i] = payoff_vec[i]*np.exp(-r*T)
        st_vec[i] = st
        vt_vec[i] = vt

    plt.hist(st_vec,bins=100)
    plt.title("Terminal stock price distribution")
    plt.xlabel("stock price")
    plt.ylabel("frequency per " + str(n) + " trials")
    plt.show()

    return price_vec


##########################################
############### INPUT ZONE ###############
##########################################

S0 = 282
K1 = 285
K2 = 315
q = 1.77e-02
r = 1.5e-02
T = 1
v0 = 0.05
sig = 0.45
k = 0.04
theta = 0.19
rho = 0

n = 1000
N = 5000

price2 = knock_simu(S0,K1,K2,q,r,T,v0,sig,k,theta,rho,n,N)
miu2 = np.mean(price2)
sig2 = np.std(price2)


##########################################
####### RELATIVE ERROR CALCULATION #######
##########################################

m1 = np.mean(price2[:-1])
m2 = np.mean(price2)
err_pct = abs(m2-m1)/m1*100


err = np.empty(0)

for N in [1000, 2000, 5000, 10000, 20000]:
    
    # pay attention to relative error convergence as N grows larger
    
    price = knock_simu(S0,K1,K2,q,r,T,v0,sig,k,theta,rho,n,N)
    miu = np.mean(price)
    sig = np.std(price)
    
    m1 = np.mean(price[:-1])
    m2 = np.mean(price)
    err_pct = abs(m2-m1)/m1*100
    err = np.append(err, err_pct)

    
    


############################# PART 3 ###############################
# Knock-out call simulation under Heston model with control variate 
####################################################################

# expect for a quicker convergence rate

covval = np.cov(price1, price2)[0][1]
cstar = -covval/sig1**2
var_new = np.var(price2) + cstar*covval
var_old = np.var(price2)
imp_pct = (var_old - var_new)/var_old*100

miu_w = np.empty(0)
miu_o = np.empty(0)
err_w = np.empty(0)
err_o = np.empty(0)

for N in [1000, 2000, 5000, 10000, 20000, 50000, 100000]:
    
    price_e = Euro_simu(S0,K1,q,r,T,v0,sig,k,theta,rho,n,N)
    price_k = knock_simu(S0,K1,K2,q,r,T,v0,sig,k,theta,rho,n,N)
    
    z = np.mean(price_e)
    
    a11 = np.mean(price_k[:-1] + cstar*(price_e-z))
    a12 = np.mean(price_k + cstar*(price_e-z))
    err_pct_w = abs(a12-a11)/a11*100
    err_w = np.append(err_w, err_pct_w)
    
    b11 = np.mean(price_k[:-1])
    b12 = = np.mean(price_k)
    err_pct_o = abs(b12-b11)/b11*100
    err_o = np.append(err_o, err_pct_o)
    
    miu_w = np.append(miu_e, np.mean(price_e))
    miu_o = np.append(miu_k, np.mean(price_k))

print(miu_w)
print(miu_o)   
print(err_w)
print(err_o)

