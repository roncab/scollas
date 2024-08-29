from spherical_collapse import *

par_EdS  = 1  ,-1 ,0
par_LCDM = 0.3,-1 ,0

def check_Eds_error():
    "Check error in delta_vir against the analytical solution in EdS." 
    print("z", "       errror [%]")
    for dmi in np.linspace(0.005,0.0015951,10):
        z, dv = solve_dv(par_EdS,dmi)
        print("{:.3f}".format(z),"  ", "{:.10f}".format(100*(dv/delta_vir_EdS-1)))


def check_nlin_error(pars):
    "Check nonlinear evolution"
    # To facilitate generalization, the dynamical equations also solve delta_m. 
    # Here we check the error between the numerical and analytical solutions for delta_m_NL. 

    dmi = 0.001
    y_nli, y_lin = ics_init(pars,dmi)
    a_grid= np.geomspace(ai,1,2000)

    sol_nli = odeint(eqs_nli, y_nli, a_grid, rtol=1e-16, atol=1e-8, printmessg=0, args=(pars,))
    
    R   = sol_nli[:,0]
    dNL = sol_nli[:,2]
    
    plt.plot(a_grid,100*(dNL/dm_R(a_grid,R,1,dmi)-1))  
    plt.xscale("log")
    plt.xlabel("a")
    plt.ylabel("error [%]")
    plt.show()

def plot_deltas(pars,dmi=0.001):
    # Visualize the linear and nonlinear evolution of delta_m.
    y_nli, y_lin = ics_init(pars,dmi)
    a_grid= np.geomspace(ai,1,2000)

    sol_nli = odeint(eqs_nli, y_nli, a_grid, rtol=1e-16, atol=1e-8, printmessg=0, args=(pars,))
    sol_lin = odeint(eqs_lin, y_lin, a_grid, rtol=1e-16, atol=1e-8, printmessg=0, args=(pars,))

    dNL = sol_nli[:,2]
    dL  = sol_lin[:,1]  
    plt.plot(a_grid,dNL,label="NL")    
    plt.plot(a_grid,dL,label="LIN")
    plt.xscale("log")
    #plt.yscale("log")    
    plt.xlabel("a")
    plt.ylabel("$\delta _m$")
    plt.legend()
    plt.show()

def test_models():
    "Examples of models that have problems, usually related to strong violation of EdS initally."
    import random
    for i in range(10):
        om = random.uniform(0, 1)
        wo = random.uniform(-2, 0)
        wa = random.uniform(-2, 2)
        p = om, wo, wa
        print(p)
        get_funcs(p)
        print("")

def plot_models():
        "Some examples of delta_vir."
        p1 = 1.0, -1.0, 0.0
        p2 = 0.3, -1.0, 0.0
        p3 = 0.2, -1.0, 0.0
        p4 = 0.4, -1.0, 0.0
        p5 = 0.3, -1.0, 0.3
        p6 = 0.3, -1.0, -0.3

        pars = [p1,p2,p3,p4,p5,p6]
        z_grid = np.linspace(0,3,300)

        for p in pars:
            dv, *_ =get_funcs(p)
            plt.plot(z_grid,dv(z_grid),label=str(p))
 #           plt.plot(z_grid,100*(dv(z_grid)/ delta_vir_EdS -1),label=str(p))

        plt.xlabel("z")
        plt.ylabel("$\delta _v$")
        plt.legend()
        plt.show()
###########################################


#Executing tests....

#check_Eds_error()

#plot_deltas(par_LCDM)

#check_nlin_error(par_EdS)

#test_models()

plot_models()
