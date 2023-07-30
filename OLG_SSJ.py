# -*- coding: utf-8 -*-
"""
Redistribution and Investment

Wendy Morrison, July 2023

"""

import numpy as np
import OLG_SSJ_ss
calibration = OLG_SSJ_ss.steadystate(noisy=False)

                    ### PARAMETERS ###
H=2
sigvec = np.arange(0,H,1)


from sequence_jacobian import simple, solved, create_model, combine

@simple
def firm(K, L, Z, alpha, delta):
    r = alpha * Z * (K / L) ** (alpha-1) - delta
    w = (1 - alpha) * Z * (K / L) ** alpha
    Y = Z * K ** alpha * L ** (1 - alpha)
    return r, w, Y

' Solve for the behavior of the current young '

@simple
def lifetimebudget_y(co_y,w,r,beta,psil,Tax,taul,gamma,H,sig_y,nh,thetao,thetay):
    
    # EIS over the life-cycle
    sig = np.ones(H)*sig_y
    sig = sig*nh**sigvec
    
    cy_y = (beta*(1+r(+1))*co_y**(-sig[H-1]))**(-1/sig[0])
    
    ly_y = ((cy_y**(-sig[0])*(1-taul)*w*thetay)/psil)**(1/gamma)
    lo_y = ((co_y**(-sig[H-1])*(1-taul)*w(+1)*thetao)/psil)**(1/gamma)
    
    lifebud_y = (thetay*w*ly_y*(1-taul)+Tax-cy_y)*(1+r(+1))+Tax+thetao*w(+1)*lo_y*(1-taul)-co_y 
    
    return lifebud_y

@simple
def lifetimebudget_o(co,w,r,beta,psil,Tax,taul,gamma,H,sig_y,nh,thetay,thetao):
    
    # EIS over the life-cycle
    sig = np.ones(H)*sig_y
    sig = sig*nh**sigvec
    
    cy_o = (beta*(1+r)*co**(-sig[H-1]))**(-1/sig[0])
    
    ly_o = ((cy_o**(-sig[0])*(1-taul)*w(-1)*thetay)/psil)**(1/gamma)
    lo_o = ((co**(-sig[H-1])*(1-taul)*w*thetao)/psil)**(1/gamma)
    
    lifebud_o = (thetay*w(-1)*ly_o*(1-taul)+Tax-cy_o)*(1+r)+Tax+thetao*w*lo_o*(1-taul)-co
    
    return lifebud_o

@simple
def household(co,co_y,w,r,beta,psil,Tax,taul,gamma,H,sig_y,nh,thetay,thetao):
    
    # EIS over the life-cycle
    sig = np.ones(H)*sig_y
    sig = sig*nh**sigvec
    
    cy = (beta*(1+r(+1))*co_y**(-sig[H-1]))**(-1/sig[0])
    
    ly = ((cy**(-sig[0])*(1-taul)*w*thetay)/psil)**(1/gamma)
    
    ay = thetay*w*ly*(1-taul)+Tax-cy
    
    lo = ((co**(-sig[H-1])*(1-taul)*w*thetao)/psil)**(1/gamma)

    
    return ay, cy, ly, lo

households = combine([household,lifetimebudget_o,lifetimebudget_y])
household_solved = households.solved(unknowns={'co':.8, 'co_y': .8},targets=['lifebud_o', 'lifebud_y'],solver='broyden_custom')

print(f"Inputs: {household_solved.inputs}")
print(f"Outputs: {household_solved.outputs}")

@simple
def targets(ay,K,pi_y,pi_o,ly,lo,L,NFA):
    asset_mrkt  = pi_y*ay - K(+1)*NFA       
    labor_mrkt = pi_y*ly + pi_o*lo - L
        
    return asset_mrkt, labor_mrkt

@simple
def fiscal(taul,L,w):
    Tax = taul*L*w*2
    return Tax

        
OLGmodel = create_model([firm,fiscal,household_solved,targets],name='OLG')

print(f"Inputs: {OLGmodel.inputs}")
print(f"Outputs: {firm.outputs}")

unknowns_ss={'psil':.3,'beta':.967}
targets_ss={'labor_mrkt':0,'asset_mrkt':0}
ss = OLGmodel.solve_steady_state(calibration, unknowns_ss, targets_ss, solver="hybr")


inputs = ['taul']
unknowns = ['K','L']
targets = ['asset_mkt','labor_mrkt']

G = OLGmodel.solve_jacobian(ss, unknowns, targets, inputs, T=4)



' Do Jacobians Manually'

J_firm = firm.jacobian(ss, inputs=['K','L'])

J_hh = household_solved.jacobian(ss,inputs=['w','r','taul','Tax'],T=4)

J_fiscal = fiscal.jacobian(ss,inputs=['taul','w','L'])

J_target = targets.jacobian(ss, inputs = ['beq','ao','K','lo','ly','ay','L'])




