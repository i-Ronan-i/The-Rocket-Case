import numpy as np
import pylab as pylab
from pylab import plot,xlabel,ylabel,title,legend,figure,subplots
import matplotlib.pyplot as plt


Rref = np.load("R.npy")
Vref = np.load("V.npy")
mref = np.load("m.npy")
tref = np.load("time.npy")

#Plot Position
plt.plot(tref, Rref/1000, "r--", label="Set point command [m]")
plt.xlabel('Time [s]')
plt.ylabel('Altitude [km]')
plt.title('Rocket Position Setpoint')
plt.xticks(np.arange(0, 190.5, step=10))
plt.yticks(np.arange(6370.0, 6440.1, step=10))
plt.legend()
plt.grid()
plt.show()

#Plot Velocity
plt.plot(tref, Vref, "g--", label="Set Point - Velocity [m/s]")
plt.xlabel('Time [s]')
plt.ylabel('Velocity [m/s]')
plt.title('Rocket Velocity Setpoint')
plt.xticks(np.arange(0, 190.5, step=10))
plt.legend()
plt.grid()
plt.show()

#Plot Mass
plt.plot(tref, mref, "b--", label="Set Point - Mass [kg]")
plt.xlabel('Time [s]')
plt.ylabel('Mass [kg]')
plt.title('Rocket Mass Setpoint')
plt.xticks(np.arange(0, 190.5, step=10))
plt.legend()
plt.grid()
plt.show()