
########################################################################################################
"""Using a set of 3 Genetic Algorithm tuned PID controllers to control the thrust of a rocket along 
a trajectory with pre-optimised position, velocity and mass."""                                     
########################################################################################################

import numpy as np
from scipy.integrate import solve_ivp, RK23, simps, BDF, RK45
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
######################################################################################################
############################################# L O A D ###############################################
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
#################################################  GA  CODE  #################################################
######################################                                   ######################################
###############################################################################################################
def create_initial(pop_num, pop, kd_min, kd_max, kp_min, kp_max, ki_min, ki_max):
    """Creates the initial population of the genetic algorithm while making sure it adheres to force constraints"""
    for s in range(pop_num):
        #Creating the random PID values
        kp_r = round(random.uniform(kp_min, kp_max), 2)
        kp_v = round(random.uniform(kp_min, kp_max), 2)
        kp_m = round(random.uniform(kp_min, kp_max), 2)
        ki_r = round(random.uniform(ki_min, ki_max), 2)
        ki_v = round(random.uniform(ki_min, ki_max), 2)
        ki_m = round(random.uniform(ki_min, ki_max), 2)
        kd_r = round(random.uniform(kd_min, kd_max), 2)
        kd_v = round(random.uniform(kd_min, kd_max), 2)
        kd_m = round(random.uniform(kd_min, kd_max), 2)

        #creating the weights 
        w_r = round(random.random(), 2)
        w_v = round(random.uniform(0, 1-w_r), 2)
        w_m = round(1 - w_r - w_v, 2)

        #Into 2-D List. Access via pop[i][j]
        pop.insert(s, [kp_r, kp_v, kp_m, ki_r, ki_v, ki_m, kd_r, kd_v, kd_m, w_r, w_v, w_m])
    return pop


def crossover(a, b):
    """Finding cut-points for crossover
    and joining the two parts of the two members
    of the population together. """
    new_a = []  #Clearing previous 
    cut_a = random.randint(1, len(a)-1) #Makes sure there is always a cut

    new_a1 = a[0 : cut_a]
    new_a2 = b[cut_a : len(b)]

    #Creates the new crossed-over list
    new_a = new_a1 + new_a2

    # Weight Check #
    ################
    #add weights and check if = 1
    #if not new weights: w_m = 1 - w_r - w_v

    return new_a


def mutate(pop, mut_prob, kd_min, kd_max, kp_min, kp_max, ki_min, ki_max): 
    """Takes current population member and add a probability chance to the PID parameters
    that it mutates via a 50:50 chance that it is reduced or increased
    by 10%. 
    However, for the weights only one is mutated if picked and the others moulded around that."""
    pop_curr = pop
    for i in range(0, len(pop_curr)):
        weight_mut = False
        for o in range(len(pop_curr[i])-3) :
            if random.random() <= mut_prob:
                if random.random() < 0.5:
                    pop_curr[i][o] = round(pop_curr[i][o] * 0.95, 2) #Maintains 2 d.p
                else :
                    pop_curr[i][o] = round(pop_curr[i][o] * 1.05, 2)
                    if pop_curr[i][0] or pop[i][1] or pop[i][2] > kp_max:
                        pop_curr[i][o] = float(kp_max) 
                    if pop_curr[i][3] or pop[i][4] or pop[i][5] > ki_max :
                        pop_curr[i][o] = float(ki_max)
                    if pop_curr[i][6] or pop[i][7] or pop[i][8] > kd_max :
                        pop_curr[i][o] = float(kd_max)
        #Weight Mutation.               
        for o in range(len(pop_curr[i]-3), len(pop_curr)):
            if random.random() <= mut_prob:
                weight_mut = True
                if random.random() < 0.5:
                    pop_curr[i][o] = round(pop_curr[i][o] * 0.95, 2) #Maintains 2 d.p
                else:
                    pop_curr[i][o] = round(pop_curr[i][o] * 1.05, 2)
            if weight_mut == True:
                if o == len(pop_curr)-3 : # w_r
                                # w_v = 1 - w_r - w_m
                    pop_curr[i][o+1] = 1 - pop_curr[i][o] - pop_curr[i][o+2]
                if o == len(pop_curr)-2 : # w_v
                                # w_m = 1 - w_v - w_r
                    pop_curr[i][o+1] = 1 - pop_curr[i][o] - pop_curr[i][o-1]
                if o == len(pop_curr)-1 : # w_m
                                # w_r = 1 - w_m
                    pop_curr[i][o-2] = 1 - pop_curr[i][o] - pop_curr[i][o-1]
                #Ensures only 1 mutation occurs
                o = len(pop_curr)
    return pop_curr


def create_next_generation(pop, pop_num, fit_val, mut_prob, kd_min, kd_max, kp_min, kp_max, ki_min, ki_max):
    """Top 20 reproduce(crossover, mutation), top 5 remain, 15 randomly created."""
    #Saves top 3 performing genomes
    pop_top = []
    for m in range(3) :
        pop_top.append(pop[m])

    #Crossover performed in top 20
    pop_cross = []
    for n in range(25):
        new_pop1 = crossover(pop[n], pop[n+1])
        pop_cross.append(new_pop1)

    #Adds all currently available members
    #Then mutates them.
    pop_new = []
    pop_premut = []
    pop_premut = pop_top + pop_cross
    pop_new = mutate(pop_premut, mut_prob, kd_min, kd_max, kp_min, kp_max, ki_min, ki_max)

    #Create random members and saves them    
    for s in range(pop_num - len(pop_new)):
        #Creating the random PID values
        kp_r = round(random.uniform(kp_min, kp_max), 2)
        kp_v = round(random.uniform(kp_min, kp_max), 2)
        kp_m = round(random.uniform(kp_min, kp_max), 2)
        ki_r = round(random.uniform(ki_min, ki_max), 2)
        ki_v = round(random.uniform(ki_min, ki_max), 2)
        ki_m = round(random.uniform(ki_min, ki_max), 2)
        kd_r = round(random.uniform(kd_min, kd_max), 2)
        kd_v = round(random.uniform(kd_min, kd_max), 2)
        kd_m = round(random.uniform(kd_min, kd_max), 2)

        #creating the weights 
        w_r = round(random.random(), 2)
        w_v = round(random.uniform(0, 1-w_r), 2)
        w_m = round(1 - w_r - w_v, 2)

        #Into 2-D List. Access via pop[i][j]
        pop_new.append([kp_r, kp_v, kp_m, ki_r, ki_v, ki_m, kd_r, kd_v, kd_m, w_r, w_v, w_m])
    return pop_new

def fit_sort(pop, fit_val):
    #This sorts the population into descending fitness (ascending order)
    switches = 1
    while switches > 0:
        switches = 0
        for i in range(len(fit_val)-1) :
            for j in range(i+1, len(fit_val)) : 
                if fit_val[i] > fit_val[j] :
                    temp = fit_val[i]
                    fit_val[i] = fit_val[j]
                    fit_val[j] = temp

                    temp2 = pop[i]
                    pop[i] = pop[j]
                    pop[j] = temp2

                    switches = switches + 1        
    #Pop list is now sorted. 
    return pop, fit_val
#####################################################################################################
##################################### I N T E G R A T I O N #########################################
#####################################################################################################
s_time = []
s_iter = 0
er_save, ev_save, em_save, R_save = [], [], [], []

pop = [[12.4, 53.6, 13.5, 655.3,123.4, 245.1, 52.6,123.5, 543.6, 0.32, 0.3, 0.34]]
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
        #er_d = np.gradient([er_save[s_iter-1], er_save[s_iter]], [s_time[s_iter-1], s_time[s_iter]])
        #ev_d = np.gradient([ev_save[s_iter-1], ev_save[s_iter]], [s_time[s_iter-1], s_time[s_iter]])
        #em_d = np.gradient([em_save[s_iter-1], em_save[s_iter]], [s_time[s_iter-1], s_time[s_iter]])

    T_pid_r = Kp_r*er + Ki_r*er_i #- Kd_r*er_d[0]
    T_pid_v = Kp_v*ev + Ki_v*ev_i #- Kd_v*ev_d[0]
    T_pid_m = Kp_m*em + Ki_m*em_i #- Kd_m*em_d[0]

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

teval = np.linspace(tref[0], tfin, int(tfin*10))
x_init = [rocket.Re, 0.0, rocket.M0]  # initial conditions
solga = solve_ivp(sysRocket, (tref[0], tfin), x_init, t_eval=teval)
#solga = RK45(sysRocket, tref[0], x_init, tfin) #, first_step=0.000001, method='LSODA', min_step=0.000001


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