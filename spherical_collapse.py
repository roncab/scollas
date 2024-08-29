import numpy as np
import math as m
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d
from scipy.optimize import brentq
from scipy.optimize import minimize_scalar
from scipy.integrate import odeint
import matplotlib.pyplot as plt

from cosmo_bck import *

# Spherical Collapse auxiliary functions.
# Mostly based on Structure Formation in the Universe, Padmanabhan.
delta_vir_EdS = 3/5 * (3/4)**(2/3)*(3*np.pi/2+1)**(2/3) #Seokcheon Lee, Physics Letters B 685, 2-3 (2010)

def t_a(a,h,om):
    return (6*G*np.pi*rho_m(a,h,om))**(-1/2)

def r(x,AI):
    "Parametric r solution."
    return AI*(1-np.cos(x))

def dm_R(a,R,Ri,dmi):
    "Nonlinear solution of delta_m(R)."
    return (1+dmi)*(R*ai/(Ri*a))**(-3) - 1

def dm_nli(T,B,x):
    "Nonlinear parametric solution as a function of x."
    return 9*(x-np.sin(x)+T/B)**2 / ( 2*(1-np.cos(x))**3 ) -1

def dm_lin(T,B,x):
    "Linear parametric solution."
    return (3/5)*(3/4)**(2/3) * (x- np.sin(x)+T/B)**(2/3)

# Routine to compute the initial condition, assumes that EdS initially.
# Based on the usual SC model, Padmanabahan's Structure Formation book, see exercise 8.1. 
def ics_init(params,dmi):
    om, wo, wa = params
    h      = 1 # Hubble parameter, dummy for dynamics.
    ti     = t_a(ai,h,om)
    Hi     = 2/(3*ti)
    dot_ai = 2*ai/(3*ti)

    ri   = 1  # Initial radius, dummy for dynamics. 
    vi   = - Hi*dmi*ri/3 # Initial peculiar velocity.
    omi  = OmegaM(ai,om,wo,wa)

    D = (1+dmi)- 1/omi*(1+vi/(Hi*ri))**2
    E = -(Hi*ri)**2 * D /2

    # Test if collapse can occur.
    if np.sign(E)>0:
        print("Initial energy is positive, no virialization willl happen.")
        return (1,1,1) ,(1,1)
            
    AI_par     = ri/(2*D) * (1+dmi) # Not initial scale factor!
    BI         = (1+dmi)/(2*Hi*np.sqrt(omi)*D**(3./2))

    xi         = np.arccos(1-ri/AI_par)
    dot_ri     = AI_par*np.sin(xi)/(BI*(1-np.cos(xi)))
    prime_ri   = dot_ri/dot_ai
    ic_dri     = prime_ri/ri
    TI         = ti - BI*(xi - np.sin(xi))

    dmiNL  = dm_nli(TI,BI,xi)
    dmiLIN = dm_lin(TI,BI,xi)

    # Test if ICs are consistent with EdS solutions.
    if dmiNL <0 or dmiLIN <0:
        print("Problem: negative ICS for Omi=", OmegaM(ai,om,wo,wa),"Om=",om,"wo=",wo,"wa=",wa)
        print(dmiNL,dmiLIN)
        return (1,1,1) ,(1,1)
    
    #        R, R',     dm
    y_nli = ri, ic_dri, dm_nli(TI,BI,xi)
    #        Rl,        dm
    y_lin = -dm_lin(TI,BI,xi)/(3*ai),  dm_lin(TI,BI,xi)

    return y_nli, y_lin

# Nonlinear system of equations
def eqs_nli(y, a, params):
    om, wo, wa = params
    R, Rp, dm= y
    dy = [ Rp,
          (1+3*wtot(a,om,wo,wa))*Rp/(2*a) - R/(2*a*a)*(OmegaM(a,om,wo,wa)*(1+dm) + OmegaDE(a,om,wo,wa)*(1+3*w(a,wo,wa))),
           -3*(Rp/R -1/a)*(1 + dm)
         ]
    return dy

# Linear system of equations
def eqs_lin(y, a, params):
    om, wo, wa = params
    R, dm = y
    dy = [ - (2 - (1+3*wtot(a,om,wo,wa))/2 )*R/a - 1/(2*a*a)*OmegaM(a,om,wo,wa)*dm ,
           -3*R
        ]
    return dy

# Routine that solves SC and compute virialization quantities
def solve_dv(params, dmi):
    om, wo, wa = params
    a_grid = np.geomspace(ai, 0.2, 5000, dtype=np.float64) # first integration grid, later ones are refined to avoid divergent delta_m

    #tables to save data for interpolating functions
    a_tab  = [] ; r_tab  = [] ; dm_tab = []
    #compute initial conditions
    y_nli, y_lin = ics_init(params,dmi)
    if y_nli[2] == 1 : # bad IC
        return -9, 0

    #Solve nonlinear evoltion until delta_m = 500, virialization is achieved before that, if
    #not, the model might be inconsistent with virialization.
    d_aux    = 0.01
    while d_aux < 500 and a_grid[-1] < 1.2:
        sol_nli = odeint(eqs_nli, y_nli, a_grid, rtol=1e-16, atol=1e-9, printmessg=0, args=(params,))
        i_sol  = len(sol_nli)-1
        i_grid = len(a_grid)-1
        if i_sol != i_grid: print("Integration problem.")
        d_aux  = sol_nli[i_sol,2]
        r      = sol_nli[i_sol,0]
        aend   = a_grid[i_grid]
        a_aux  = np.delete(a_grid, i_grid) #remove the last point to avoid repetition
        a_tab.extend(a_aux)
        r_tab.extend(sol_nli[0:i_grid,0])
        dm_tab.extend(sol_nli[0:i_grid,2])
        y_nli  = sol_nli[i_sol] # next set o initial conditions
        a_grid = np.linspace(aend,aend*1.03, 2000, dtype=np.float64) # next refined grid for solution
        
    if a_grid[-1] > 1.19 and sol_nli[-1,2] < 80 :
        return -8, 0 # dm does not grow enough to virialize
     
    a_tab  = np.array(a_tab) ; r_tab  = np.array(r_tab); dm_tab = np.array(dm_tab)

    # now solve the linear evolution to get delta_v and growth function
    a_tab_lin  = np.geomspace(ai, aend, 2000, dtype=np.float64)
    sol_lin    = odeint(eqs_lin, y_lin, a_tab_lin, rtol=1e-16, atol=1e-8, printmessg=0,args=(params,))

    #intepolating functions
    dm_NL       = interp1d(a_tab, dm_tab, kind='cubic')
    dm_LI       = interp1d(a_tab_lin, sol_lin[:,1], kind='cubic')
    rad         = InterpolatedUnivariateSpline(a_tab, r_tab, k=4) 
    r_tab_minus = [i*(-1) for i in r_tab]
    rad_minus   = interp1d(a_tab, r_tab_minus, kind='cubic')

    #Find turn around
    ata  = minimize_scalar(rad_minus,bounds=(0.1,aend),method='bounded').x
    rta  = rad(ata) # usefull for testing

    #Radius derivatives
    Drad  = rad.derivative(1)
    DDrad = rad.derivative(2)

    DDa = lambda a: - (1+3*wtot(a,om,wo,wa))/(2*a)

    #Virialization equation
    vir_eq  = lambda a : (Drad(a)/rad(a))**2 +(DDrad(a)+DDa(a)*Drad(a))/rad(a)

    aend=a_tab[-2] # avoid the last value, not included in the interpolation
    if(np.sign(vir_eq(ata)) == np.sign(vir_eq(aend))):
        print("Virialization equations has no solution.")
        return -8, 0 # Lack of virialization
    
    #  Virialization solution
    avir = brentq(vir_eq,ata,aend)
    dvir = dm_LI(avir)

    return  1/avir-1, dvir

def growth(params):
    "Computes only the linear growth"
    om, wo, wa = params
    aend      = 1.01
    a_tab     = np.geomspace(ai, aend, 2000, dtype=np.float64)
    _, y_lin  = ics_init(params,0.001)
    sol_lin   = odeint(eqs_lin, y_lin, a_tab, rtol=1e-16, atol=1e-8, printmessg=0,args=(params,))
    dm_LI     = interp1d(a_tab, sol_lin[:,1], kind='cubic')
    dtot_LI   = lambda a : dm_LI(a)
    
    g_a     = lambda a : dtot_LI(a)/dtot_LI(1.)
    g_inter = InterpolatedUnivariateSpline(a_tab, g_a(a_tab), k=4)
    dgda    = g_inter.derivative(1)
    f_a     = lambda a: a*dgda(a)/g_a(a)
    gamma_a = lambda a: np.log(f_a(a))/np.log(OmegaM(a,om,wo,wa))

    g_z      = lambda z: g_a(1./(1+z))
    f_z      = lambda z: f_a(1./(1+z))
    gamma_z  = lambda z: gamma_a(1./(1+z))

    return g_z, f_z, gamma_z
                                                                   
def get_funcs(params):
    "Routine that provides interpolating function for dv."
    om, wo, wa = params
    tab_zv = []; tab_dv = []
    dm_run = 0.007 # for most models, this value should give the first virialization around z=3

    bad_model_ini = 0 # flag to report a bad model at start
    bad_model_run = 0 # flag to report a bad model evolution (usually lack of virialization)
    zvir, dvir = solve_dv(params,dm_run)

    if zvir == -9: # Bad IC.
        bad_model_ini = 1
        print("Bad initial conditions, parameters with problem om=",om,"wo=",wo,"wa=",wa)
        return 0, 0, 0, 0, bad_model_run, bad_model_ini
  
    if zvir == -8: # No virialization
        bad_model_run = 1
        print("Lack of virialization, parameters with problem om=",om,"wo=",wo,"wa=",wa)
        return 0, 0, 0, 0, bad_model_run, bad_model_ini

    while zvir < 3 and zvir != -8 and zvir!= -9: #assure table begins at least at zv=3                                   
        dm_run = dm_run * 1.1
        zvir, dvir = solve_dv(params,dm_run)

    while zvir > 0 and zvir != -8 and zvir!= -9:
        zvir, dvir = solve_dv(params,dm_run)

        if zvir != -9 and zvir != -8:
            tab_zv.append(zvir)
            tab_dv.append(dvir)
        else: # in case of lack of virialization or bad IC at some point, complete tables with the last value
            if (zvir == -0.9):
                bad_model_ini = 1
                print("Bad initial conditions.")
                print("Parameters with problem om=",om,"wo=",wo,"wa=",wa)         
            if (zvir == -0.8):
                bad_model_run = 1
                print ("Lack of virialization beyond z", tab_zv[-1])   
                print("Parameters with problem om=",om,"wo=",wo,"wa=",wa)
            zvir = tab_zv[-1]
            dvir = tab_dv[-1]
            while zvir > 0:
                zvir = zvir - 0.03
                tab_zv.append(zvir)
                tab_dv.append(dvir)

        dm_run = dm_run*0.98

    # Get the growth function 
    g_z, f_z, gamma_z = growth(params)
    
    dv_z = interp1d(tab_zv, tab_dv, kind='cubic') #delta_vir 
          
    return dv_z, g_z, f_z, gamma_z, bad_model_ini, bad_model_run
########################################################################