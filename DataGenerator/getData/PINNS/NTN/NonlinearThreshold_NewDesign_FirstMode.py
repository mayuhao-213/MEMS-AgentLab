# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 20:48:54 2018

@author: god_m
"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab

from scipy.optimize import fsolve

import math

import scipy.io as spio

from sympy.core.symbol import symbols
from sympy.solvers.solveset import nonlinsolve

#mat = spio.loadmat('Modeshapes_6X500.mat', squeeze_me=True)
w_t = 8e-6
l_t = 800e-6

numberofsmallelements = 10000
length_x = np.zeros(numberofsmallelements)
modeshape_unnormalized = np.zeros(numberofsmallelements)
second_derivative = np.zeros(numberofsmallelements)
first_derivative = np.zeros(numberofsmallelements)
#modeshape1 = np.zeros(numberofsmallelements)

beta = 4.730041 # beta for first mode
#beta = 7.853205 # beta for second mode
alpha_n = (np.sin(beta) - np.sinh(beta)) / (np.cosh(beta) - np.cos(beta))

for jj in range(numberofsmallelements):
    length_x[jj] = jj* l_t / numberofsmallelements
    modeshape_unnormalized[jj] = -(-np.sin(beta * length_x[jj] / l_t) + np.sinh(beta * length_x[jj] / l_t) + alpha_n * (-np.cos(beta * length_x[jj] / l_t) + np.cosh(beta * length_x[jj] / l_t)))

for jj in range(numberofsmallelements):
    second_derivative[jj] = -1 / np.max(modeshape_unnormalized) * (beta ** 2) * (np.sin(beta * length_x[jj] / l_t) + np.sinh(beta * length_x[jj] / l_t) + alpha_n * (np.cos(beta * length_x[jj] / l_t) + np.cosh(beta * length_x[jj] / l_t)))

for jj in range(numberofsmallelements):
    first_derivative[jj] = -1 / np.max(modeshape_unnormalized) * beta * (-np.cos(beta * length_x[jj] / l_t) + np.sinh(beta * length_x[jj] / l_t) + alpha_n * (np.sin(beta * length_x[jj] / l_t) + np.cosh(beta * length_x[jj] / l_t)))

modeshape1 = modeshape_unnormalized / np.max(modeshape_unnormalized)
#second_derivative = second_derivative_un / np.max(np.abs(second_derivative_un))

dx = length_x[1] - length_x[0]

m_coef_b = np.sum(modeshape1 ** 2 * dx / l_t)
k_coef_b = np.sum(second_derivative ** 2 * dx / l_t) 
k_coef_b3 = np.sum(first_derivative ** 2 * dx / l_t)
    

#ModeShapes = mat['phi']   ## all mode shapes
#modeshape1 = ModeShapes[:,0]    ## first mode shape

#length_x = mat['yy']
dx = length_x[1] - length_x[0]

#overlap_start_length = 6e-06
#overlap_stop_length = 300e-6

#start_index = length_x.tolist().index(overlap_start_length)
#stop_index = length_x.tolist().index(overlap_stop_length)
#coef = np.sum(modeshape1[start_index : stop_index])

electrode_length = 700e-6
electrode_width = 20e-6

w_c = 10e-6
l_c = 20e-6

#coef = 0.66175786 * 4300 ## COMSOL simulated

## Define mechanical properties of resonator ##

E = 169e9 # Young's modulus
rho = 2330
t = 25e-6
m_coef = 1



#w_c = 5e-6
#l_c = 150e-6

Q = 20000

k_tt = k_coef_b / 12 * E * t * ((w_t / l_t) ** 3)
k_t3 = k_coef_b3 * E * t * w_t / (l_t ** 3)

#k_c = k_coef_b / 12 * E * t * ((w_c / l_c) ** 3)
k_t = k_tt
#k_t = 1 / (1 / k_tt + 2 / k_c)

#Mass = rho * t * ((w_t * l_t + 2 * w_c * l_c) * m_coef_b + (6e-6 + 15e-6) * 40e-6 * m_coef / 2)
Mass = rho * (t * w_t * l_t * m_coef_b + electrode_length * electrode_width * t + w_c * l_c * t *2)

c = math.sqrt(Mass * k_t) / Q

omega_0 = math.sqrt(k_t / Mass)
freq_0 = omega_0 / (2 * math.pi)
## end defining mechanical properties ## 

## Define electrical properties of resonator ##

eps_0 = 8.85e-12
V = 6.4
d = 2e-6
trans_factor = eps_0 * V * electrode_length * t  / (d ** 2) 

k_e = 2 * trans_factor * V / d
k_e3 = 4 * trans_factor * V / (d ** 3)

## end defining electrical properties


SMALL_SIZE = 14
MEDIUM_SIZE = 14
BIGGER_SIZE = 14

axis_linewidth = 1
data_linewidth = 2
datafit_linewidth = 2


#string = f"Noise floor is \n {noise_floor_round} $\mu$Hz/Hz$^1$$^{/}$$2$"
#string = f"Noise floor: \n {noise_floor_round} $\mu$Hz/Hz$^1$$^/$$^2$"

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = 'Times New Roman:italic'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

number_of_sim = 360
number_of_drive = 1

freq = np.zeros((number_of_sim, number_of_drive))
m_c = np.zeros((number_of_sim, number_of_drive)) ## motional current
phi = np.zeros(number_of_sim)
label_fig = ["" for x in range(number_of_drive)]

for ii in range(number_of_drive):

    vac = 5e-3 + 0.1e-3 * ii        
    force_ac = vac * trans_factor
    vac_round = np.round(vac / 1e-3 , 2)
    label_fig[ii] = "vac = {} mV".format(vac_round)

    for i in range(number_of_sim):
        phi[i] = i / number_of_sim * (160 / 180) * math.pi + 10 / 180 * math.pi
#phi = math.pi / 2
        x, y = symbols('x, y', reals=True)

        solutions =  nonlinsolve([-Mass * (x ** 2) * y + (k_t - k_e) * y + (k_t3 - k_e3) * (y ** 3) * 3 / 4 - force_ac * math.cos(phi[i]), 
                                  c * x * y - force_ac * math.sin(phi[i])], 
                                 [x, y])
        for j in range(len(solutions)):
            sol = solutions.args[j]
            if np.abs(sol[0]) > omega_0 / 2 and np.abs(sol[0]) < omega_0 * 3 / 2 and sol[0] > 0:
                freq[i, ii] = sol[0] / (2 * math.pi)
                m_c[i, ii] = sol[1] * freq[i, ii] * 2 * math.pi * trans_factor / 1e-9
            else:
                pass
            
#plt.plot(freq, m_c, label = label_fig)
#plt.xlabel("Frequency (Hz)")
#plt.ylabel("Motional current (nA)")
#plt.legend()    

mpl_fig = plt.figure()
ax = mpl_fig.add_subplot(111)

line1, = ax.plot(freq[:,0], m_c[:,0], lw=data_linewidth, color = 'tab:red', label = label_fig[0])
line2, = ax.plot(freq[:,1], m_c[:,1], lw=data_linewidth, color = 'tab:blue', label = label_fig[1])
line3, = ax.plot(freq[:,2], m_c[:,2], lw=data_linewidth, color = 'tab:green', label = label_fig[2])
line4, = ax.plot(freq[:,3], m_c[:,3], lw=data_linewidth, color = 'tab:orange', label = label_fig[3])
line5, = ax.plot(freq[:,4], m_c[:,4], lw=data_linewidth, color = 'tab:purple', label = label_fig[4])

#ax.plot(Time_stamp, model(Time_stamp, *popt1), lw=data_linewidth, color = 'r-')

#ax.set_title("My Plot Title")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Motional current (nA)")

ax.tick_params(which='both', direction='in')
ax.grid(linestyle='--', linewidth=0.5)
#ax.axhline(y=noise_floor, linewidth=1.5, color='r', label='Noise floor = ')
ax.spines['left'].set_linewidth(axis_linewidth)
ax.spines['right'].set_linewidth(axis_linewidth)
ax.spines['top'].set_linewidth(axis_linewidth)
ax.spines['bottom'].set_linewidth(axis_linewidth)

ax.legend()
