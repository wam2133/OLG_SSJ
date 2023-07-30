# -*- coding: utf-8 -*-
"""
Solves for the steady state of the OLG model

"""

import numpy as np
import utils

                    ### PARAMETERS ###
K=1.6
L=1                    
delta = .3/K # 20 % investment share # 
alpha = .3
H = 2 # Number of generations #
sig_y = 1.4
nh = .9 # set to 1 for homothetic lifetime savings
gamma = 1.
taul=.3
sigvec = np.arange(0,H,1)
thetay = 1.9
thetao = .1
pi_o = .5
pi_y = .5
NFA = 1-16/25


' First, solve for Z that ensures K=K, L=1, and r = .03 '

Z = (.05+delta)/(alpha*(K / L) ** (alpha-1))

' Solve for steady state w and Y'

def firm_ss(K, L, Z, alpha, delta):
    r = alpha * Z * (K / L) ** (alpha-1) - delta
    w = (1 - alpha) * Z * (K / L) ** alpha
    Y = Z * K ** alpha * L ** (1 - alpha)
    return r, w, Y

w = firm_ss(K, L, Z, alpha, delta)[1]
Y = firm_ss(K, L, Z, alpha, delta)[2]
r = .05



' Governments budget clears '
Tax = taul*L*w*2

' Solve for psia, psil, and beta to satify hh problem, clear labor markets, and clear asset markets '

def find_util(x_guess):
    psil=x_guess[0]
    beta=x_guess[1]    
    
    sig = np.ones(H)*sig_y
    sig = sig*nh**sigvec
    
    def lifetimebudget_ss(co_guess):
        
        co = co_guess[0]

        cy = (beta*(1+r)*co**(-sig[H-1]))**(-1/sig[0])
    
        ly = ((cy**(-sig[0])*(1-taul)*w*thetay)/psil)**(1/gamma)
        lo = ((co**(-sig[H-1])*(1-taul)*w*thetao)/psil)**(1/gamma)
    
        lifebud = (thetay*w*ly*(1-taul)+Tax-cy)*(1+r)+Tax+thetao*w*lo*(1-taul)-co 
    
        return np.array([lifebud])
    
    co=utils.broyden_solver(lifetimebudget_ss,np.array([.8]),noisy=True)[0][0]
    
    cy = (beta*(1+r)*co**(-sig[H-1]))**(-1/sig[0])
    
    ly = ((cy**(-sig[0])*(1-taul)*w*thetay)/psil)**(1/gamma)
    lo = ((co**(-sig[H-1])*(1-taul)*w*thetao)/psil)**(1/gamma)
    
    ay = thetay*w*ly*(1-taul)+Tax-cy
    
    labor_mrkt = L - pi_o*lo - pi_y*ly
    asset_mrkt  = pi_y*ay - K*NFA  

    return np.array([labor_mrkt, asset_mrkt])

psil = utils.broyden_solver(find_util,np.array([.3,.99]),noisy=True)[0][0]
beta = utils.broyden_solver(find_util,np.array([.3,.99]),noisy=True)[0][1]

sig = np.ones(H)*sig_y
sig = sig*nh**sigvec
    
def lifetimebudget_ss(co_guess):
        
    co = co_guess[0]

    cy = (beta*(1+r)*co**(-sig[H-1]))**(-1/sig[0])
    
    ly = ((cy**(-sig[0])*(1-taul)*w*thetay)/psil)**(1/gamma)
    lo = ((co**(-sig[H-1])*(1-taul)*w*thetao)/psil)**(1/gamma)
    
    lifebud = (thetay*w*ly*(1-taul)+Tax-cy)*(1+r)+Tax+thetao*w*lo*(1-taul)-co 
    
    return np.array([lifebud])
    
co=utils.broyden_solver(lifetimebudget_ss,np.array([.8]),noisy=True)[0][0]
    
cy = (beta*(1+r)*co**(-sig[H-1]))**(-1/sig[0])
    
ly = ((cy**(-sig[0])*(1-taul)*w*thetay)/psil)**(1/gamma)
lo = ((co**(-sig[H-1])*(1-taul)*w*thetao)/psil)**(1/gamma)
    
ay = thetay*w*ly*(1-taul)+Tax-cy


def steadystate(psil=psil,beta=beta,noisy=True):

    ss = {'thetao':thetao,'thetay':thetay,'pi_o':pi_o,'pi_y':pi_y ,'K':K,'L':1,'delta':delta,'H':2,'alpha':alpha,'sig_y':sig_y,'nh':.9,
          'gamma':gamma,'Tax':Tax,'taul':taul,'co':co,'cy':cy,'ly':ly,'lo':lo,'ay':ay,
          'r':r,'w':w,'Y':Y,'Z':Z,'bequests':0,'NFA':NFA}

    return ss

# Test Walras Law #

calibration = {'thetao': 0.1, 'thetay': 1.9, 'pi_o': 0.5, 'pi_y': 0.5, 'K': 1.6, 'L': 1, 'delta': 0.18749999999999997, 'H': 2, 'alpha': 0.3, 'sig_y': 1.4, 'nh': 0.9, 'gamma': 1.0, 'Tax': 0.5319999999999999, 'taul': 0.3, 'co': 1.7476602802635783, 'cy': 1.6233880298203545, 'ly': 1.9023585540600496, 'lo': 0.09764144761560888, 'ay': 1.1520000010308595, 'r': 0.05, 'w': 0.8866666666666665, 'Y': 1.2666666666666666, 'Z': 1.100085263739135, 'bequests': 0, 'NFA': 0.36}


















    





    





