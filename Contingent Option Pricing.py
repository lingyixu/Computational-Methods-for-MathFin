

###########################################################################
# Contingent option: payoff is contingent on the dynamics of another option
# Vanilla option VS contingent option
# Influence of option correlation change
# Influence of contingent condition change
# Given sigma1 = 0.2, sigma2 = 0.15, rho = 0.95
###########################################################################


# 1. vanilla call pricing using quadrature
import scipy.integrate.quadrature as quad

S0 = 271.0
K1 = 260.0
t1 = 1.0
r = 0
sigma1 = 20.0

def myfunc1(x):
    return np.exp(-r*t1)*(x-K1)*sc.norm.pdf(x, S0, sigma1)

van_price, van_error = quad(myfunc1, K1, S0+5*sigma1)



# 2. contingent call pricing using quadrature
from scipy.stats import multivariate_normal
from scipy.integrate import dblquad as dquad

K2 = 250
t2 = .5
sigma2 = 15
rho = .95

mv = multivariate_normal(mean=[S0,S0], cov=[[1,rho],[rho,1]])

def myfunc2(x1, x2):
    return mv.pdf([x1,x2])*np.exp(-r*t1)*(x2-K1)

con_price, _ = dquad(myfunc2, K1, S0+10*sigma1, lambda x2: S0-10*sigma2, lambda x2: K2)



# 3. call price with different correlations
rho_list = [0.8, 0.5, 0.2]
price_list1 = {}

for rho in rho_list:
    mv = multivariate_normal(mean=[S0,S0], cov=[[1,rho],[rho,1]])
    def myfunc3(x1, x2):
        return mv.pdf([x1,x2])*np.exp(-r*t1)*(x2-K1)
    con_price, _ = dquad(myfunc3, K1, S0+10*sigma1, lambda x2: S0-10*sigma2, lambda x2: K2)
    price_list1[rho] = con_price


    
# 4. call price with different contingent conditions
K2_list = [240, 230, 220]
price_list2 = {}
for K2 in K2_list:
    def myfunc4(x1, x2):
        return mv.pdf([x1,x2])*np.exp(-r*t1)*(x2-K1)
    con_price, _ = dquad(myfunc3, K1, S0+10*sigma1, lambda x2: S0-10*sigma2, lambda x2: K2)
    price_list2[K2] = con_price


    
# 5. try contingent option price with extreme parameter values
rho_new = 0
K2_new = 100000

mv = multivariate_normal(mean=[S0,S0], cov=[[1,rho_new],[rho_new,1]])

def myfunc5(x1, x2):
    return mv.pdf([x1,x2])*np.exp(-r*t1)*(x2-K1)

con_price_new, _ = dquad(myfunc5, K1, S0+10*sigma1, lambda x2: S0-10*sigma2, lambda x2: K2_new)

