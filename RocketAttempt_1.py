import numpy as np
from scipy.integrate import solve_ivp, RK23, odeint, simps
from scipy.interpolate import PchipInterpolator
import pylab as pylab
from pylab import plot,xlabel,ylabel,title,legend,figure,subplots
import matplotlib.pyplot as plt
import random
import time

""" Using a set of 3 Genetic Algorithm tuned PID controllers to control the thrust of a rocket along 
a trajectory with pre-optimised position, velocity and mass. """

###### Define rocket flying parameters parameters #####
class Rocket:
    GMe = 3.986004418 * 10**14  # Earth gravitational constant [m^3/s^2]
    Re = 6371.0 * 1000  # Earth Radius [m]
    g0 = 9.80665  # Gravitational acceleration on Earth surface [m/s^2]

    def __init__(self):
        self.M0 = 5000  # Initial total mass [kg]
        self.Mc = 0.4  # Initial Propellant mass over total mass
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
#####
######################################################################################################
#Load from reference files and interpolate paths.
Rref = np.load("R.npy")
Vref = np.load("V.npy")
mref = np.load("m.npy")
tref = np.load("time.npy")
tfin = tref[-1]
Rfun = PchipInterpolator(tref, Rref)
Vfun = PchipInterpolator(tref, Vref)
mfun = PchipInterpolator(tref, mref)

###############################################################################################################
#################################################             #################################################
#################################################  B E G I N  #################################################
######################################                                   ######################################
###############################################################################################################

#pop = [[kp_r, kp_v, kp_m, ki_r, ki_v, ki_m, kd_r, kd_v, kd_m, w_r, w_v, w_m]]
pop = [[210.0, 342.0, 524.0, 631.0, 512.0, 134.0, 535.0, 485.0, 183.0, 0.33, 0.33, 0.34]]

s=0
Kp_r = pop[s][0]
Kp_v = pop[s][1]
Kp_m = pop[s][2]
Ki_r = pop[s][3]
Ki_v = pop[s][4]
Ki_m = pop[s][5]
Kd_r = pop[s][6]
Kd_v = pop[s][7]
Kd_m = pop[s][8]
W_r = pop[s][9]
W_v = pop[s][10]
W_m = pop[s][11]


#####################################################################################################
##################################### I N T E G R A T I O N #########################################
#####################################################################################################
s_time = []
s_iter = 0
er_save, ev_save, em_save, R_save = [], [], [], []

def sysRocket(t, x):
    global s_iter, s_time
    s_time.append(t)
    flag = False
    # State Variables
    R = x[0]
    V = x[1]
    m = x[2]

    if R - rocket.Re < 0:
        R = rocket.Re
        crash = True
    if m < rocket.M0*rocket.Mc:
        m = rocket.M0*rocket.Mc
        flag = True
    elif m > rocket.M0:
        m = rocket.M0
        flag = True
    if abs(V) > 1e3:
        flag = True
        if V > 0:
            V = 1e3
        else:
            V = -1e3

    r = Rfun(t)
    v = Vfun(t)
    mf = mfun(t)

    #Global save values for storing errors for integration and differentiation
    global er_save, ev_save, em_save, R_save
    R_save.append(R)
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
        er_i = simps([er_save[s_iter-1], er_save[s_iter]], x=[s_time[s_iter-1], s_time[s_iter]])
        ev_i = simps([ev_save[s_iter-1], ev_save[s_iter]], x=[s_time[s_iter-1], s_time[s_iter]])
        em_i = simps([em_save[s_iter-1], em_save[s_iter]], x=[s_time[s_iter-1], s_time[s_iter]])
           #differentiate ( last error to current error ) over time (last time to current time)
        er_d = np.gradient([er_save[s_iter-1], er_save[s_iter]], [s_time[s_iter-1], s_time[s_iter]])
        ev_d = np.gradient([ev_save[s_iter-1], ev_save[s_iter]], [s_time[s_iter-1], s_time[s_iter]])
        em_d = np.gradient([em_save[s_iter-1], em_save[s_iter]], [s_time[s_iter-1], s_time[s_iter]])

    T_pid_r = Kp_r*er + Ki_r*er_i - Kd_r*er_d[0]
    T_pid_v = Kp_v*ev + Ki_v*ev_i - Kd_v*ev_d[0]
    T_pid_m = Kp_m*em + Ki_m*em_i - Kd_m*em_d[0]

    T = W_r * T_pid_r + W_v * T_pid_v + W_m * T_pid_v
    if T > rocket.Tmax:
        T = rocket.Tmax
        flag = True
    elif T < 0.0:
        T = 0.0
        flag = True
    #####    

    rho = rocket.air_density(R - rocket.Re)
    drag = 0.5 * rho * V ** 2 * rocket.Cd * rocket.area
    g = rocket.GMe / R ** 2
    g0 = rocket.g0
    Isp = rocket.Isp

    #increment for PID calculations
    s_iter = s_iter + 1
    dxdt = [0,0,0]

    dxdt[0] = V
    dxdt[1] = (T - drag) / m - g
    dxdt[2] = - T / g0 / Isp
    return dxdt[0], dxdt[1], dxdt[2]

teval = np.linspace(0, tfin, int(tfin*10))
x_ini = [rocket.Re, 0.0, rocket.M0]  # initial conditions
solga = solve_ivp(sysRocket, [tref[0], tfin], x_ini, dense_output=True, t_eval=teval)#, first_step=0.000001, method='LSODA', min_step=0.000001
y_out = solga.y[0, :]
t_out = solga.t


#Plotting code
plt.plot(tref, Rref, "r--", label="Set point command (m)")
plt.plot(s_time, R_save, label = "System Output - N") 
plt.xlabel('Time (s)')
# Set the y axis label of the current axis.
plt.ylabel('Thrust (N)')
# Set a title of the current axes.
#plt.title('System Response to Varying Step Inputs')
plt.title('System Response - Thrust')

#plt.xticks(np.arange(0, tref[-1], step=2))

# show a legend on the plot
plt.legend()
# Display a figure.
plt.grid()
plt.show()