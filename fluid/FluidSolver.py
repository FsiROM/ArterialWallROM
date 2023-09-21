from __future__ import division, print_function
import os
import sys
import argparse
import outputConfiguration as config
from thetaScheme import perform_partitioned_implicit_trapezoidal_rule_step, perform_partitioned_implicit_euler_step
import numpy as np
import tubePlotting
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from output import writeOutputToVTK
import precice
from precice import action_write_initial_data, action_write_iteration_checkpoint, \
    action_read_iteration_checkpoint
import time

nln_veloc = True
law = 'strs-strain'
bC = 'reflecting'

# physical properties of the tube
r0 = 1 / np.sqrt(np.pi)  # radius of the tube
a0 = r0**2 * np.pi  # cross sectional area

if law == 'cubic-law':
    u0 = 3.  # mean velocity
    ampl = 0.818  # amplitude of varying velocity
    frequency = .8  # frequency of variation

elif law == 'strs-strain':
    u0 = 5.
    ampl = 1.
    frequency = 2.
else:
    u0 = 10.
    ampl = 3.
    frequency = 10  # frequency of variation

t_shift = 0  # temporal shift of variation
p0 = 0  # pressure at outlet
kappa = 100
rho = 1
p0 = 0.
r0 = 1 / np.sqrt(np.pi)  # radius of the tube
E = 10000  # elasticity module
c_mk = np.sqrt(E / 2 / r0)  # wave speed

L = 10  # length of tube/simulation domain
N = 100
dx = L / kappa
# helper function to create constant cross section
save_iter_data = False


pressure0 = p0 * np.ones(N + 1)


def pres_in(t):

    ksi = .01
    return 300 * np.exp(-(t)**2/(2*ksi*2))


if nln_veloc:
    if law == 'cubic-law':
        freq = 2.
        pres = 120.
        c = -0.001
        ntt = 1200
        dt = .08
        b = 0.
        d2 = 3.
        e = -.02

    elif law == 'strs-strain':
        freq = .9
        #freq = 2.
        pres = 120.
        c = -.002
        ntt = 1200
        dt = .1
        b = 0.
        d2 = 3.
        e = -.02
    else:
        freq = 1.
        pres = 250.
        c = -1.
        ntt = 600
        dt = .01
        b = 0.
        d2 = 3.
        e = -.02


    a = -1.
    d1 = -1


    def p(t): return pres * np.cos(freq * t)

    def d(t): return d1 + d2 * p(t)
    def v_dot(u, v, t): return c * u**3 + b * u**2 + a * u + d(t) + e * v

    input_t = np.arange(ntt)*dt

    from scipy.integrate import solve_ivp
    def f(t, y): return np.array([y[1], v_dot(y[0], y[1], t)])
    sol = solve_ivp(f, [0, ntt*dt], np.array([10., 0]), t_eval=input_t)


def velocity_in(t):
    if nln_veloc:
        if law == 'cubic-law':
            return sol.y[0][int(t/dt)]/130.+6.
        elif law == 'strs-strain':
            #return sol.y[0][int(t/dt)]/60. + 6.
            return sol.y[0][int(t/dt)]/60. + 4.
        else:
            return sol.y[0][int(t/dt)] + 11.
    else:
        return u0 + ampl * np.sin(frequency * t * np.pi)


def crossSection0(N):
    return a0 * np.ones(N + 1)


def energy_flux(crossSection, pressure, velocity):

    tmp = (.5 * rho * crossSection * (velocity**3) +
           pressure * crossSection * velocity)
    return tmp[-1] - tmp[0]


def fluid_kinetic_energy(crossSection, velocity):

    return .5 * rho * crossSection * velocity**2


def ksi(sect):

    vect = (2*rho*(c_mk**2)*sect) - 2 * np.sqrt(a0*sect)*(2*rho*c_mk**2-p0)

    return vect


def tube_energy(crossSection):

    return ksi(crossSection)


def energy_density(crossSection, velocity):

    return tube_energy(crossSection) + fluid_kinetic_energy(crossSection, velocity)


def total_energy(crossSection, velocity):

    return np.trapz(energy_density(crossSection, velocity), dx=dx)


def dEdt(energ, old_energ, dt):

    return (energ - old_energ)/dt


parser = argparse.ArgumentParser()
parser.add_argument("configurationFileName", help="Name of the xml precice configuration file.",
                    nargs='?', type=str, default="../precice-config.xml")
parser.add_argument(
    "--enable-plot", help="Show a continuously updated plot of the tube while simulating.", action='store_true')
parser.add_argument("--write-video", help="Save a video of the simulation as 'writer_test.mp4'. \
                    NOTE: This requires 'enable_plot' to be active!", action='store_true')

try:
    args = parser.parse_args()
except SystemExit:
    print("")
    print("Did you forget adding the precice configuration file as an argument?")
    print("Try '$ python FluidSolver.py precice-config.xml'")
    quit()

plotting_mode = config.PlottingModes.VIDEO if args.enable_plot else config.PlottingModes.OFF
if args.write_video and not args.enable_plot:
    print("")
    print("To create a video it is required to enable plotting for this run.")
    print("Please supply both the '--enable-plot' and '--write-video' flags.")
    quit()
writeVideoToFile = True if args.write_video else False

#print("Plotting Mode: {}".format(plotting_mode))

#print("Starting Fluid Solver...")

#print("N: " + str(N))

#print("Configure preCICE...")
interface = precice.Interface("Fluid", args.configurationFileName, 0, 1)
#print("preCICE configured...")

dimensions = interface.get_dimensions()

velocity = velocity_in(0) * np.ones(N + 1)
velocity_old = velocity_in(0) * np.ones(N + 1)
pressure = p0 * np.ones(N + 1)
pressure_old = p0 * np.ones(N + 1)

crossSectionLength = a0 * np.ones(N + 1)
crossSectionLength_old = a0 * np.ones(N + 1)

if plotting_mode == config.PlottingModes.VIDEO:
    fig, ax = plt.subplots(1)
    if writeVideoToFile:
        FFMpegWriter = manimation.writers['imagemagick']
        metadata = dict(title='PulseTube')
        writer = FFMpegWriter(fps=15, metadata=metadata)
        writer.setup(fig, "writer_test.mp4", 100)

meshID = interface.get_mesh_id("Fluid-Nodes-Mesh")
crossSectionLengthID = interface.get_data_id("CrossSectionLength", meshID)
pressureID = interface.get_data_id("Pressure", meshID)

vertexIDs = np.zeros(N + 1)
grid = np.zeros([N + 1, dimensions])

grid[:, 0] = np.linspace(0, L, N + 1)  # x component
grid[:, 1] = 0  # y component, leave blank

vertexIDs = interface.set_mesh_vertices(meshID, grid)

t = 0

#print("Fluid: init precice...")
# preCICE defines timestep size of solver via precice-config.xml
precice_dt = interface.initialize()

if interface.is_action_required(action_write_initial_data()):
    interface.write_block_scalar_data(pressureID, vertexIDs, pressure)
    interface.mark_action_fulfilled(action_write_initial_data())

interface.initialize_data()

if interface.is_read_data_available():
    crossSectionLength = interface.read_block_scalar_data(
        crossSectionLengthID, vertexIDs)

#crossSectionLength_old = np.copy(crossSectionLength)
# initialize such that mass conservation is fulfilled
velocity_old = velocity_in(
    0) * crossSectionLength_old[0] * np.ones(N + 1) / crossSectionLength_old


time_it = 0
iters = 0
iters_tot = 0

# =====Initialising energy storing arrays=========
energy_balance = np.array([0.])
energy_fluxes = np.array([energy_flux(crossSectionLength, pressure, velocity)])
dEdts = np.array([0])
old_enrg = total_energy(crossSectionLength, velocity)
conv_ids = [1]
crossSectionLength_old_saved = crossSectionLength_old.copy()

times = []
times_fl = []

t2 = time.time()
while interface.is_coupling_ongoing():
    # When an implicit coupling scheme is used, checkpointing is required
    if interface.is_action_required(action_write_iteration_checkpoint()):
        interface.mark_action_fulfilled(action_write_iteration_checkpoint())


    t0 = time.time()
    velocity, pressure, success = perform_partitioned_implicit_euler_step(
        velocity_old, pressure_old, crossSectionLength_old, crossSectionLength, dx, precice_dt, velocity_in(
            t + precice_dt), custom_coupling=True, pres=pres_in(t+precice_dt), law=law, bC=bC)
    t1 = time.time()
    times_fl.append(t1 - t0)
    #print("\n  ---   --  PRESSURE IS  , ", velocity_in(t), " -- \n")

    interface.write_block_scalar_data(pressureID, vertexIDs, pressure)
    interface.advance(precice_dt)
    crossSectionLength = interface.read_block_scalar_data(
        crossSectionLengthID, vertexIDs)

    # ========Compute energy balance at the fixed point iterations========
    #enrg = total_energy(crossSectionLength, velocity)
    #dedt = dEdt(enrg, old_enrg, precice_dt)
    #enrg_flx = energy_flux(crossSectionLength, pressure, velocity)

    #energy_fluxes = np.append(energy_fluxes, enrg_flx)
    #dEdts = np.append(dEdts, dedt)

    #energy_balance = np.append(energy_balance, enrg_flx+dedt)
    conv_ids.append(0)

    #np.save("energy_balance.npy", energy_balance)
    #np.save("energ_flux.npy", energy_fluxes)
    #np.save("energ_dedt.npy", dEdts)
    #np.save("energy_convergence_ids.npy", np.array([conv_ids]))

    if save_iter_data:
        writeOutputToVTK(iters_tot, "out_fluid_interm_", dx, datanames=["velocity", "pressure", "diameter"], data=[
            velocity, pressure, crossSectionLength])
    iters_tot += 1
    iters += 1

    # i.e. not yet converged
    if interface.is_action_required(action_read_iteration_checkpoint()):
        crossSectionLength_old_saved = crossSectionLength.copy()
        interface.mark_action_fulfilled(action_read_iteration_checkpoint())
    else:  # converged, timestep complete
        t3 = time.time()
        times.append(t3-t2)
        t += precice_dt

        # convergence_id in energy
        conv_ids[-1] = 1
        #old_enrg = enrg

        if plotting_mode is config.PlottingModes.VIDEO:
            tubePlotting.doPlotting(
                ax, crossSectionLength_old, velocity_old, pressure_old, dx, t)
            if writeVideoToFile:
                writer.grab_frame()
            ax.cla()
        velocity_old = np.copy(velocity)
        pressure_old = np.copy(pressure)
        crossSectionLength_old = np.copy(crossSectionLength_old_saved)
        iters_arr = iters * np.ones_like(pressure_old)
        writeOutputToVTK(time_it, "out_fluid_", dx, datanames=["velocity", "pressure", "diameter", "iters"], data=[
            velocity_old, pressure_old, crossSectionLength_old, iters_arr])
        time_it += 1
        iters = 0
        t2 = time.time()

#print("Exiting FluidSolver")

if plotting_mode is config.PlottingModes.VIDEO and writeVideoToFile:
    writer.finish()

np.save("times2.npy", np.array(times))
np.save("times_fl.npy", np.array(times_fl))
interface.finalize()
