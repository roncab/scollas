#Basic cosmological functions
import numpy as np
import math as m

# Physical constants ----------------------------
G     = 6.674184e-11   # Nm^2/kg^2
pc    = 3.08568025e16  # m
kpc   = 1e3 * pc
Mpc   = 1e3 * kpc
Msun  = 1.98847e30     # kg
c_vel = 3 * 10 ** 8    # m/s
Gyear = 365*24*60*60   # s

# Basic Cosmological functions -------------------
ai = 0.001

def Ho(h):
    return 100*h/kpc # 1/s

def rho_c_o(h):
    return 3/(8*np.pi*G) * Ho(h)**2 # kg/m^3

def rho_c_o_Msun(h):
    return rho_c_o(h)*Mpc**3/Msun # Msun/Mpc^3

def w(a,wo,wa):
    a = np.array(a)
    return wo + (1-a)*wa

def fw(a,wo,wa):
    a = np.array(a)
    return a**(-3*(1+wa+wo))*np.exp(3*wa*(a-1))

def Ehub(a,om,wo,wa):
    a = np.array(a)
    return (om*a**(-3) + (1-om)*fw(a))**(1/2)

def OmegaM(a,om,wo,wa):
    a = np.array(a)
    return om*a**(-3)/(om*a**(-3) + (1-om)*fw(a,wo,wa))

def OmegaDE(a,om,wo,wa):
    a = np.array(a)
    return 1-OmegaM(a,om,wo,wa)

def wtot(a,om,wo,wa):
    a = np.array(a)
    return w(a,wo,wa)*OmegaDE(a,om,wo,wa)

def rho_m(a,h,om):
    a = np.array(a)
    return rho_c_o(h) * om * a **(-3)

def rho_m_Msun(a,h,om):
    a = np.array(a)
    return rho_c_o_Msun(h) * om * a **(-3)

def rho_de(a,h,om,wo,wa):
    a = np.array(a)
    return rho_c_o(h) * (1-om) * fw(a,wo,wa)




################################################
#Testing area
if __name__ == "__main__":
    print(rho_m_Msun(1.,0.7,0.3)/10**11)