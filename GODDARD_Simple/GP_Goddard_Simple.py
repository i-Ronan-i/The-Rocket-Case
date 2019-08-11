import numpy as np
from scipy.interpolate import PchipInterpolator
###############################  S Y S T E M - P A R A M E T E R S  ####################################################

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

obj = Rocket()
Nstates = 3
Ncontrols = 1

top_end_stop = 80  # [km]
bottom_end_stop = 0.0  # [km]

tref = np.load("time.npy")
total_time_simulation = tref[-1]

######################################################################################################
Rref = np.load("R.npy")
Vref = np.load("V.npy")
mref = np.load("m.npy")
tref = np.load("time.npy")
tfin = tref[-1]


Rfun = PchipInterpolator(tref, Rref)
Vfun = PchipInterpolator(tref, Vref)
mfun = PchipInterpolator(tref, mref)
#############################################INTEGRATION###############################################

x_ini = [obj.Re, 0.0, obj.M0]  # initial conditions

def sys(t, x):
    # State Variables
    R = x[0]
    V = x[1]
    m = x[2]

    if R < 0 or np.isnan(R):
        R = obj.Re
        flag = True
    if np.isinf(R) or R > obj.Re+80e3:
        R = obj.Re + 80e3
        flag = True
    if m < obj.M0*obj.Mc or np.isnan(m):
        m = obj.M0*obj.Mc
        flag = True
    elif m > obj.M0 or np.isinf(m):
        m = obj.M0
        flag = True
    if abs(V) > 1e3 or np.isinf(V):
        if V > 0:
            V = 1e3
            flag = True
        else:
            V = -1e3
            flag = True


    r = Rfun(t)
    v = Vfun(t)
    mf = mfun(t)

    er = r - R
    ev = v - V
    em = mf - m
    dxdt = np.zeros(Nstates)
    # print("Fr: ", fTr(er, et, evr, evt, em))

    rho = obj.air_density(R - obj.Re)

    drag = 0.5 * rho * V ** 2 * obj.Cd * obj.area
    g = obj.GMe / R ** 2
    g0 = obj.g0
    Isp = obj.Isp
    T = fT(er, ev, em)


    if abs(fT(er, ev, em)) > obj.Tmax or np.isinf(fT(er, ev, em)):
        T = obj.Tmax
        flag = True

    elif fT(er, ev, em) < 0.0 or np.isnan(fT(er, ev, em)):
        T = 0.0
        flag = True

    dxdt[0] = V
    dxdt[1] = (T - drag) / m - g
    dxdt[2] = - T / g0 / Isp
    return dxdt

tin = 0.0
teval = np.linspace(0, tfin, int(tfin*4))
if flag_offdesign is True:
    x_ini = xnew_ini
    tin = change_time
    teval = t_evals2
sol = solve_ivp(sys, [tin, tfin], x_ini, dense_output=True, t_eval=teval)
y1 = sol.y[0, :]
y2 = sol.y[1, :]
y3 = sol.y[2, :]
tt = sol.t

if sol.t[-1] != tfin:
    flag = True

r = Rfun(tt)
v = Vfun(tt)
m = mfun(tt)

err1 = (r - y1)/(obj.Re+50*1e3)
err2 = (v - y2)/np.sqrt(obj.GMe/obj.Re)
err3 = (m - y3)/obj.M0

