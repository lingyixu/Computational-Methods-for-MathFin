# -*- coding: utf-8 -*-

# MF 796 - Assignment 4
# Date: 2019-02-17


# (3)
# the total number of securities is 100
G = np.matrix([np.ones(100), np.append(np.ones(17),np.zeros(100-17))])
cov_mat = np.matrix(cov.values)
C = cov_mat

# get pseudoinverse of C
C_inv = np.linalg.pinv(C)
P = G*C_inv*(G.T)
inverse_mat = P.I

# (4)
c = np.matrix([[1],[0.1]])
a = 1
R = np.matrix(return_daily_data.mean().values)
lam = inverse_mat*(G*C_inv*(R.T)-2*a*c)
weight = 0.5/a*C_inv*(R.T-(G.T)*lam)

