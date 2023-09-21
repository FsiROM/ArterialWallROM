from __future__ import division, print_function

import os
from re import M
import sys
import argparse
import numpy as np
from numpy import save
from numpy import savez_compressed
import precice
from precice import action_write_initial_data, action_read_iteration_checkpoint, \
    action_write_iteration_checkpoint
from scipy.interpolate import RBFInterpolator
from rom_am import ROM, POD, DMDc #TODO this should be called from the rom_am package 
from sklearn.decomposition import KernelPCA
from solid_rom import solid_ROM
from scipy.optimize import root
import time

law = 'strs-strain'

r0 = 1 / np.sqrt(np.pi)  # radius of the tube
a0 = r0**2 * np.pi  # cross sectional area
tau = 10**10  # timestep size, set it to a large value to enforce tau from precice_config.xml
N = 100  # number of elements in x direction
p0 = 0  # pressure at outlet
L = 10  # length of tube/simulation domain
if law == 'cubic-law':
    from scipy.optimize import minimize
    from scipy.interpolate import UnivariateSpline
    x_ = np.array([0., 20.,  50., 100., 250., 1000., 2000., 3000., 4000.])
    y = np.array([a0, 1.005, 1.01, 1.011, 1.05,  1.15, 1.24, 1.43, 1.63])
    y_neg = np.array([0.995, 0.99, 0.989, 0.95, 0.85, 0.76, 0.758, 0.752])

    x_ = np.sort(np.append(x_, - x_[1::]))
    y = np.sort(np.append(y, y_neg))

    def had_law(pres):
        res = np.empty_like(pres)

        res[(pres < 50)+(pres > -50)] = (1.005-1) / \
            50*pres[(pres < 50)+(pres > -50)]+1.
        a = (1.012-1.005)/(300-50)
        b = 1.012 - a*300.
        res[pres > 50] = a*pres[pres > 50]+b
        b = 2 - 1.012 + a * 300
        res[pres < -50] = a*pres[pres < -50]+b

        return res

    x_ = np.append(np.linspace(0, 140., 6),  np.linspace(160., 300., 6))
    y = had_law(x_)
    y_neg = 2 - y[1::]

    x_ = np.sort(np.append(x_, - x_[1::]))
    y = np.sort(np.append(y, y_neg))

    spl = UnivariateSpline(x_, y, s=1e-21)
elif law == 'strs-strain':
    def strs(eps):
        res = np.empty_like(eps)

        res[(eps < .002)+(eps > -.002)] = 25/.002 * \
            eps[(eps < .002)+(eps > -.002)]
        a = (30-25)/(.004-.002)
        b = 30 - a*.004
        res[eps > .002] = a*eps[eps > .002]+b
        b = -30 + a * .004
        res[eps < -.002] = a*eps[eps < -.002]+b

        return res

E = 10000  # elasticity module
c_mk = np.sqrt(E / 2 / r0)  # wave speed
training_final_time = 1.  # Length of time period on which to train the ROM
rom_use = True
rom_meth = 0  # 0 for POD-POD use, 1 for kPCA-POD, 2 for isomap, 3 for new_interp
partial_data = False
save_rom = False
load_rom = True

u0 = 10.  # mean velocity
ampl = 3  # amplitude of varying velocity
frequency = 10  # frequency of variation

SolidFirst = False
first_approx = False


def velocity_in(t):
    return u0 + ampl * np.sin(frequency * t * np.pi)


def crossSection0(N):
    return a0 * np.ones(N + 1)


#print("Starting Solid Solver...")

parser = argparse.ArgumentParser()
parser.add_argument("configurationFileName", help="Name of the xml config file.", nargs='?', type=str,
                    default="precice-config.xml")

try:
    args = parser.parse_args()
except SystemExit:
    print("")
    print("Did you forget adding the precice configuration file as an argument?")
    print("Try '$ python SolidSolver.py precice-config.xml'")
    quit()

#print("N: " + str(N))

#print("Configure preCICE...")
interface = precice.Interface("Solid", args.configurationFileName, 0, 1)
#print("preCICE configured...")

dimensions = interface.get_dimensions()

pressure = p0 * np.ones(N + 1)
crossSectionLength = a0 * np.ones(N + 1)

meshID = interface.get_mesh_id("Solid-Nodes-Mesh")
crossSectionLengthID = interface.get_data_id("CrossSectionLength", meshID)
pressureID = interface.get_data_id("Pressure", meshID)

vertexIDs = np.zeros(N + 1)
grid = np.zeros([N + 1, dimensions])

grid[:, 0] = np.linspace(0, L, N + 1)  # x component
grid[:, 1] = 0  # np.linspace(0, config.L, N+1)  # y component, leave blank

vertexIDs = interface.set_mesh_vertices(meshID, grid)

t = 0

#print("Solid: init precice...")


def solid_fom(pres, pres0, cmk, section):

    if law == 'cubic-law':
        return spl(pres)
    elif law == 'strs-strain':
        def fun(x): return ((pres*x)) - strs((x-r0)/r0)
        res = root(fun, crossSectionLength)
        return np.pi * res.x**2

    else:
        return section * (
            (pres0 - 2.0 * cmk ** 2) ** 2 / (pres - 2.0 * cmk ** 2) ** 2)


# preCICE defines timestep size of solver via precice-config.xml
precice_dt = interface.initialize()

if interface.is_action_required(action_write_initial_data()):
    interface.write_block_scalar_data(
        crossSectionLengthID, vertexIDs, crossSectionLength)
    interface.mark_action_fulfilled(action_write_initial_data())

interface.initialize_data()

if not SolidFirst:
    if interface.is_read_data_available():
        pressure = interface.read_block_scalar_data(pressureID, vertexIDs)

crossSection0 = crossSection0(pressure.shape[0] - 1)
pressure0 = p0 * np.ones_like(pressure)

if rom_use:
    trained = False
    section_data = crossSection0.reshape((-1, 1))
    pressure_data = pressure0.reshape((-1, 1))
    sect_pred = np.zeros_like(section_data.copy())
    pres_pred = np.zeros_like(pressure_data.copy())
    if rom_meth == 0:
        dmdc_data = pressure0.copy().reshape((-1, 1))

iters_cnt = 0
T = 1
iters = np.array([])
old_pressure = pressure0.reshape((-1, 1))
init_pres = pressure0.reshape((-1, 1))
save_pressure = pressure0.copy()
previous_pres = pressure0.copy()
residual_ = 0.
times_sol = []
times = []
tr_time = []
t2 = time.time()
while interface.is_coupling_ongoing():
    # When an implicit coupling scheme is used, checkpointing is required
    if interface.is_action_required(action_write_iteration_checkpoint()):
        interface.mark_action_fulfilled(action_write_iteration_checkpoint())

    if t < training_final_time:
        #print("\n  ---   --  iter IS  , ", iters_cnt, " -- \n")
        #print("\n  ---   --  PRESSURE IS  , ", pressure[0], " -- \n")
        t0 = time.time()
        crossSectionLength = solid_fom(
            pressure, pressure0, c_mk, crossSection0)
        t1 = time.time()
        times_sol.append(t1-t0)
        #print("\n  ---   --  3ta section , ", crossSectionLength[0], " -- \n")

        if rom_use:
            if (not partial_data):
                pressure_data = np.hstack(
                    (pressure_data, pressure.reshape((-1, 1))))
                section_data = np.hstack(
                    (section_data, crossSectionLength.reshape((-1, 1))))
    else:

        if rom_use:
            if not trained:

                print("\n===================================================\n")
                print("\n===========ROM Training===============\n")


                t4 = time.time()

                if load_rom:
                    import pickle
                    with open('sol_rom_saved.pkl', 'rb') as inp:
                        sol_rom = pickle.load(inp)

                else:
                    np.save("../nln_elastic_law_strstrain_nln_veloc/pres_TRAIN_DATA.npy", pressure_data)
                    np.save("../nln_elastic_law_strstrain_nln_veloc/sec_TRAIN_DATA.npy", section_data)
                    sol_rom = solid_ROM()
                    sol_rom.train(pressure_data, section_data, quad_ = False, ridge = False,
                                    kernel='thin_plate_spline', degree = 1, norm_regr = True, rank_disp=9, rank_pres=3, alpha = 1e-6)
                t5 = time.time()
                tr_time.append(t5-t4)
            
                trained = True
                if save_rom:
                    import pickle

                    with open('sol_rom_saved.pkl', 'wb') as outp:
                        pickle.dump(sol_rom, outp, pickle.HIGHEST_PROTOCOL)

                print("\n===========ROM is trained===============\n")
                print("\n===================================================\n")

            print("\nSolid ROM Prediction Regime: \n")

            t0 = time.time()
            crossSectionLength = sol_rom.pred(
                pressure.reshape((-1, 1))).ravel()
            t1 = time.time()
            times_sol.append(t1 - t0)
            
            # print("\n  ---   --  3ta section , ",
            #      crossSectionLength[0], " -- \n")

            #pres_pred = np.hstack((pres_pred, pressure.reshape((-1, 1))))
            #sect_pred = np.hstack(
            #    (sect_pred, crossSectionLength.reshape((-1, 1))))

            #np.save("predicted_sect.npy", sect_pred[:, 1::])
            #np.save("predicted_pres.npy", pres_pred[:, 1::])

        else:

            t0 = time.time()
            crossSectionLength = solid_fom(
                pressure, pressure0, c_mk, crossSection0)
            t1 = time.time()
            times_sol.append(t1 - t0)

    interface.write_block_scalar_data(
        crossSectionLengthID, vertexIDs, crossSectionLength)
    precice_dt = interface.advance(precice_dt)

    """
    if t < training_final_time and not trained:
        section_data = np.hstack(
            (section_data, interface.read_block_scalar_data(
                crossSectionLengthID, vertexIDs).reshape((-1, 1))))
    """

    save_pressure = pressure.copy()
    pressure = interface.read_block_scalar_data(pressureID, vertexIDs)

    residual_ = np.linalg.norm(previous_pres - pressure) / \
        np.linalg.norm(previous_pres)
    iters_cnt += 1

    # i.e. not yet converged
    if interface.is_action_required(action_read_iteration_checkpoint()):
        interface.mark_action_fulfilled(action_read_iteration_checkpoint())
        previous_pres = pressure.copy()
    else:
        t3 = time.time()
        times.append(t3 - t2)
        iters = np.append(iters, iters_cnt)
        t += precice_dt
        iters_cnt = 0
        if rom_use:
            if partial_data:
                pressure_data = np.hstack(
                    (pressure_data, save_pressure.reshape((-1, 1))))
                section_data = np.hstack(
                    (section_data, crossSectionLength.reshape((-1, 1))))
            T += 1
            if first_approx:
                dmdc_data = np.hstack(
                    (dmdc_data, save_pressure.reshape((-1, 1))))
        old_pressure = save_pressure.reshape((-1, 1))
        t2 = time.time()

#print("Exiting SolidSolver")

interface.finalize()

np.save("times_sol.npy", np.array(times_sol))
np.save("times.npy", np.array(times))
np.save("tr_time.npy", np.array(tr_time))
