---
title: ArterialWallROM
permalink: tutorials-elastic-tube-1d.html
keywords: OpenFOAM, python, ROM-FOM
summary: Applying the ROM-FOM approach on a 1D vessel FSI case. Based on ...
---

## Setup

We want to simulate the internal flow in a flexible tube as shown in the figure below (image from [1]).

![FSI3 setup](images/tutorials-elastic-tube-1d-setup.png)

The flow is assumed to be incompressible flow and gravity is neglected. Due to the axisymmetry, the flow can be described using a quasi-two-dimensional continuity and momentum equations. The motivation and exact formulation of the equations that we consider can be found in [2].

The following parameters have been chosen:

- Length of the tube: L = 10
- Inlet velocity: $$ v_{inlet} = 10 + 3 sin (10 \pi t) $$
- Initial cross sectional area = 1
- Initial velocity: v = 10
- Initial pressure: p = 0
- Fluid density: $$ \rho = 1 $$
- Young modulus: E = 10000

Additionally the solvers use the parameters `N = 100` (number of cells), `tau = 0.01` (dimensionless timestep size), `kappa = 100` (dimensionless structural stiffness) by default. These values can be modified directly in each solver.

## ROM Setup



## Running the Simulation

Open two separate terminals and start each participant by calling the respective run script. Only serial run is possible:

```bash
cd fluid-python
./run.sh
```

and

```bash
cd solid-python
./run.sh
```

## Post-processing

![Elastic tube animation](images/tutorials-elastic-tube-1d-animation.gif)

The results from each simulation are stored in each `fluid-<participant>/output/` folder. You can visualize these VTK files using the provided `plot-diameter.sh` script

```bash
./plot-diameter.sh
```

which will try to visualize the results from both fluid cases, if available. This script calls the more flexible `plot-vtk.py` Python script, which you can use as

```bash
python3 plot-vtk.py <quantity> <case>/output/<prefix>
```

Note the required arguments specifying which quantity to plot (`pressure`, `velocity` or `diameter`) and the name prefix of the target vtk files.

For example, to plot the diameter of the fluid-python case using the default prefix for VTK files, `plot-diameter.sh` executes:

```bash
python3 plot-vtk.py diameter fluid-python/output/out_fluid_
```

![FSI3 setup](images/tutorials-elastic-tube-1d-diameter.png)

## References

[1] B. Gatzhammer. Efficient and Flexible Partitioned Simulation of Fluid-Structure Interactions. Technische Universitaet Muenchen, Fakultaet fuer Informatik, 2014.

[2] J. Degroote, P. Bruggeman, R. Haelterman, and J. Vierendeels. Stability of a coupling technique for partitioned solvers in FSI applications. Computers & Structures, 2008.

[3] M. Mehl, B. Uekermann, H. Bijl, D. Blom, B. Gatzhammer, and A. van Zuijlen.
Parallel coupling numerics for partitioned fluid-structure interaction simulations. CAMWA, 2016.  
