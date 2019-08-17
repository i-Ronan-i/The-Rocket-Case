
########################################################################################################
"""Using a set of 3 Genetic Algorithm tuned PID controllers to control the thrust of a rocket along 
a trajectory with pre-optimised position, velocity and mass."""                                     
########################################################################################################

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
tfin = tref[-1]
Rfun = PchipInterpolator(tref, Rref)
Vfun = PchipInterpolator(tref, Vref)
mfun = PchipInterpolator(tref, mref)
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
    W_r = pop[9]
    W_v = pop[10]
    W_m = pop[11]
    R_path = []
    V_path = []
    m_path = []
    R_diff = []
    V_diff = []
    m_diff = []
    er_save = []
    ev_save = []
    em_save = []
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

        T = W_r * T_pid_r + W_v * T_pid_v + W_m * T_pid_v
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
#################################################             ################################################
#################################################  GA  CODE  #################################################
######################################                                   #####################################
##############################################################################################################
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
        for o in range(len(pop_curr[i])-3, len(pop_curr[i])):
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


def fitness(pop):
    """Calculates the fitness values of each member in pop[population]
    Also sets the simulation time in timesteps."""
    fit_val = []
    for s in range(len(pop)):
        r_path, v_path, m_path = selfRocketSolve(x_init, pop[s])

        r_error = 0.0
        v_error = 0.0
        m_error = 0.0
        for i in range(len(tref)):
            r_error = r_error + abs(Rref[i] - r_path[i])
            v_error = v_error + abs(Vref[i] - v_path[i])
            m_error = m_error + abs(mref[i] - m_path[i])

        err_val = r_error + v_error + m_error
        fit_val.insert(s, err_val)
    return fit_val
#####################################################################################################
##################################### M A I N   C O D E   B O D Y ###################################
#####################################################################################################
start_time = time.time()
x_init = [rocket.Re, 0.0, rocket.M0]  # initial conditions
pop_num = 60    #How large the initial population is
pop = []
mut_prob = 0.08  #probability for mutation set here
iteration_max = 1 #Total number of iterations and generations set here
#Minimum and maximum PID coefficient gains.
kd_min, kd_max = 0, 1000
kp_min, kp_max = 0, 1000
ki_min, ki_max = 0, 1000


iteration = 0
while iteration < iteration_max:
    if iteration == 0:
        pop = create_initial(pop_num, pop, kd_min, kd_max, kp_min, kp_max, ki_min, ki_max)
        fit_val = fitness(pop)
        iteration = iteration + 1

    if iteration < iteration_max and iteration > 0:
        pop, fit_val = fit_sort(pop, fit_val)
        pop = create_next_generation(pop, pop_num, fit_val, mut_prob, kd_min, kd_max, kp_min, kp_max, ki_min, ki_max)
        fit_val = fitness(pop)
        iteration = iteration + 1


#This is the final section with the top solution being chosen and used
#Final simulation run
pop, fit_val = fit_sort(pop, fit_val)
print("Top overall Coefficients are: ", pop[0])
print("Fitness value of top performing member: ", round(fit_val[0], 4))
print("Time elapsed: ", (time.time()-start_time)/60, " minutes.")

r_path, v_path, m_path = selfRocketSolve(x_init, pop[0])

#Plot Position
plt.plot(tref, Rref, "r--", label="Set point command [m]")
plt.plot(tref, r_path, label = "System Position [m]") 
plt.xlabel('Time [s]')
plt.ylabel('Altitude [m]')
plt.title('System Response - Position')
#plt.xticks(np.arange(0, tref[-1], step=2))
plt.legend()
plt.grid()
plt.show()

#Plot Velocity
plt.plot(tref, Vref, "r--", label="Set Point - Velocity [m/s]")
plt.plot(tref, V_path, label = "System Velocity [m/s]") 
plt.xlabel('Time [s]')
plt.ylabel('Velocity [m/s]')
plt.title('System Response - Velocity')
#plt.xticks(np.arange(0, tref[-1], step=2))
plt.legend()
plt.grid()
plt.show()

#Plot Mass
plt.plot(tref, mref, "r--", label="Set Point - Mass [kg]")
plt.plot(tref, m_path, label = "System Mass [kg]") 
plt.xlabel('Time [s]')
plt.ylabel('Mass [kg]')
plt.title('System Response - Mass')
#plt.xticks(np.arange(0, tref[-1], step=2))
plt.legend()
plt.grid()
plt.show()