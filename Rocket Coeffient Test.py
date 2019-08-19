import numpy as np
from scipy.integrate import solve_ivp, RK23, simps, odeint
from scipy.interpolate import PchipInterpolator
import pylab as pylab
from pylab import plot,xlabel,ylabel,title,legend,figure,subplots
import matplotlib.pyplot as plt
import random
import time

#################################################################################################
####################### Define rocket flying parameters parameters ##############################
class Rocket:
    GMe = 3.986004418 * 10**14  # Earth gravitational constant [m^3/s^2]
    Re = 6371.0 * 1000  # Earth Radius [m]
    g0 = 9.80665  # Gravitational acceleration on Earth surface [m/s^2]

    def __init__(self):
        self.M0 = 5000  # Initial total mass [kg]
        self.Mc = 0.4  # Final mass over total mass
        self.Cd = 0.2  # Drag Coefficient [-]
        self.area = 10  # area [m2]
        self.Isp = 300.0  # Isp [s]
        self.max_thrust = 2  # maximum thrust to initial weight ratio
        self.Tmax = self.max_thrust * self.M0 * self.g0

    def air_density(self, h):
        beta = 1/8500.0  # scale factor [1/m]
        rho0 = 1.225  # kg/m3
        return rho0*np.exp(-beta*h)

rocket = Rocket()
######################################################################################################
############################################# L O A D ###############################################
#Load from reference files and interpolate paths.
Rref = np.load("R.npy")
Vref = np.load("V.npy")
mref = np.load("m.npy")
tref = np.load("time.npy")
##############################################################################################################
#################################### R O C K E T   S O L V E R ###############################################
##############################################################################################################
def selfRocketSolve(x_init, pop):
    #initialise variables
    #Grab the pop PID values
    Kp_r = pop[0]
    Kp_v = pop[1]
    Kp_m = pop[2]
    Ki_r = pop[3]
    Ki_v = pop[4]
    Ki_m = pop[5]
    Kd_r = pop[6]
    Kd_v = pop[7]
    Kd_m = pop[8]
    #W_r = pop[9]
    #W_v = pop[10]
    #W_m = pop[11]
    R_path = []
    V_path = []
    m_path = []
    R_diff = []
    V_diff = []
    m_diff = []
    er_save = []
    ev_save = []
    em_save = []
    err_save = []
    evv_save = []
    emm_save = []
    dRdt = [0,0,0]
    #Main Loop for solve
    for i in range(len(tref)):
        t = tref[i]
        if t == 0:
            R = x_init[0]
            V = x_init[1]
            m = x_init[2]
        else:
            #Integration for state values
            #integrate between tref[i-1] and tref[i]
            x_in= [R_path[i-1], V_path[i-1], m_path[i-1]]
            def sysMinistep(t,x):
                # State Variables
                R = x[0]
                V = x[1]
                m = x[2]

                #Conditions of flight constraint
                if R - rocket.Re < 0:
                    R = rocket.Re
                    ground = True
                if m < rocket.M0*rocket.Mc:
                    m = rocket.M0*rocket.Mc
                    nofuel = True
                elif m > rocket.M0:
                    m = rocket.M0
                    tooheavy = True
                if abs(V) > 1e3:
                    speedmax = True
                    if V > 0:
                        V = 1e3
                    else:
                        V = -1e3 

                rho = rocket.air_density(R - rocket.Re)
                drag = 0.5 * rho * V ** 2 * rocket.Cd * rocket.area
                g = rocket.GMe / R ** 2

                dxdt = [0, 0, 0]
                dxdt[0] = V
                dxdt[1] = (T - drag) / m - g
                dxdt[2] = - T / g0 / Isp
                return dxdt[0], dxdt[1], dxdt[2]
            #Custom time ODE solver to get next values
            solga = solve_ivp(sysMinistep, [tref[i-1], tref[i]], x_in, t_eval=[tref[i-1], tref[i]])
            
            R = solga.y[0][1]
            V = solga.y[1][1]
            m = solga.y[2][1]
        
        #Conditions of flight constraint
        if R - rocket.Re < 0:
            R = rocket.Re
            ground = True
        if m < rocket.M0*rocket.Mc:
            m = rocket.M0*rocket.Mc
            nofuel = True
        elif m > rocket.M0:
            m = rocket.M0
            tooheavy = True
        if abs(V) > 1e3:
            speedmax = True
            if V > 0:
                V = 1e3
            else:
                V = -1e3        

        #Store values to a list for further use.
        R_path.append(R)
        V_path.append(V)
        m_path.append(m)

        #Collect reference values at time tref[i]
        r = Rref[i]
        v = Vref[i]
        mf = mref[i]
        #Error values are calculated and stored in lists
        er = r - R
        er_save.append(er)
        ev = v - V
        ev_save.append(ev)
        em = mf - m
        em_save.append(em)

        ##### PID Thrust Calculations #####
        if t == 0:
            er_i = 0.0
            ev_i = 0.0
            em_i = 0.0
            er_d = [0.0]
            ev_d = [0.0]
            em_d = [0.0]
        else:
                #integrate ( last error to current error ) over time (last time to current time)
            er_i = simps([er_save[i-1], er_save[i]], x=[tref[i-1], tref[i]])
            ev_i = simps([ev_save[i-1], ev_save[i]], x=[tref[i-1], tref[i]])
            em_i = simps([em_save[i-1], em_save[i]], x=[tref[i-1], tref[i]])
            #differentiate ( last error to current error ) over time (last time to current time)
            er_d = np.gradient([er_save[i-1], er_save[i]], [tref[i-1], tref[i]])
            ev_d = np.gradient([ev_save[i-1], ev_save[i]], [tref[i-1], tref[i]])
            em_d = np.gradient([em_save[i-1], em_save[i]], [tref[i-1], tref[i]])

        T_pid_r = Kp_r*er + Ki_r*er_i - Kd_r*er_d[0]
        T_pid_v = Kp_v*ev + Ki_v*ev_i - Kd_v*ev_d[0]
        T_pid_m = Kp_m*em + Ki_m*em_i - Kd_m*em_d[0]

        #T = W_r * T_pid_r + W_v * T_pid_v + W_m * T_pid_m
        T = T_pid_r + T_pid_v + T_pid_m
        if T > rocket.Tmax:
            T = rocket.Tmax
            thrust_maxed = True
        elif T < 0.0:
            T = 0.0
            flag = True
        #####    

        rho = rocket.air_density(R - rocket.Re)
        drag = 0.5 * rho * V ** 2 * rocket.Cd * rocket.area
        g = rocket.GMe / R ** 2
        g0 = rocket.g0
        Isp = rocket.Isp


        dRdt[0] = V
        dRdt[1] = (T - drag) / m - g
        dRdt[2] = - T / g0 / Isp

        R_diff.append(dRdt[0])
        V_diff.append(dRdt[1])
        m_diff.append(dRdt[2])
    return R_path, V_path, m_path
##############################################################################################################
##############################################################################################################
#pop = [979.25, 968.03, 89.4, 508.96, 140.99, 576.34, 38.42, 591.68, 853.42] #no weights halfway results
#pop = [1000.0, 1000.0, 89.18, 726.86, 842.59, 985.53, 42.04, 591.68, 916.54] #no weights generation 28

#pop = [803.3, 1000.0, 137.42, 1000.0, 282.22, 106.79, 351.84, 609.02, 936.74, 1.0, 0.0, 0.0] # / error by weight factor
    #Fitness value of top performing member:  1,328,556.213
    #Time elapsed:  489.50  minutes.
#pop = [949.88, 822.21, 620.6, 616.12, 429.04, 40.69, 41.46, 48.38, 997.5, 0.84, 0.16, 0.0] # Original
    #Fitness value of top performing member:  1,017,078.2062
    #Time elapsed:  492.20  minutes.
pop = [1000.0, 1000.0, 103.24, 974.06, 796.46, 985.53, 42.04, 591.68, 916.54] #weight removed
    #Fitness value of top performing member:  554116.1454
    #Time elapsed:  493.4378672281901  minutes.

#time elapsed in creation = 545.34 minutes.
x_init = [rocket.Re, 0.0, rocket.M0]  # initial conditions

r_path, v_path, m_path = selfRocketSolve(x_init, pop)

r_error = 0.0
v_error = 0.0
m_error = 0.0
for i in range(len(tref)):
    r_error = r_error + abs(Rref[i] - r_path[i])/10
    v_error = v_error + abs(Vref[i] - v_path[i])
    m_error = m_error + abs(mref[i] - m_path[i])

err_val = r_error + v_error + m_error
fit_val = err_val

print("Top overall Coefficients are: ", pop[0])
print("Fitness value of top performing member: ", round(fit_val, 4))

for i in range(len(r_path)):
    r_path[i] = r_path[i]/1000

#Plot Position
plt.plot(tref, Rref/1000, "r--", label="Set point - Position [m]")
plt.plot(tref, r_path, label = "Rocket Position [m]") 
plt.xlabel('Time [s]')
plt.ylabel('Altitude [km]')
plt.title('Rocket Position Over Time')
plt.xticks(np.arange(0, 190.5, step=10))
plt.yticks(np.arange(6370.0, 6440.1, step=10))
plt.legend()
plt.grid()
plt.show()

#Plot Velocity
plt.plot(tref, Vref, "r--", label="Set Point - Velocity [m/s]")
plt.plot(tref, v_path, label = "Rocket Velocity [m/s]") 
plt.xlabel('Time [s]')
plt.ylabel('Velocity [m/s]')
plt.title('Rocket Velocity Over Time')
plt.xticks(np.arange(0, 190.5, step=10))
plt.legend()
plt.grid()
plt.show()

#Plot Mass
plt.plot(tref, mref, "r--", label="Set Point - Mass [kg]")
plt.plot(tref, m_path, label = "System Mass [kg]") 
plt.xlabel('Time [s]')
plt.ylabel('Mass [kg]')
plt.title('Rocket Mass Over Time')
plt.xticks(np.arange(0, 190.5, step=10))
plt.legend()
plt.grid()
plt.show()