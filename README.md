## Setup

This is a test case of the ROM-FOM approach [4](more reference coming soon about that) on a preCICE tutorial : [elastic-tube-1d](https://github.com/precice/tutorials/tree/master/elastic-tube-1d)

## ROM Setup

You can choose whether to use the solid ROM or not in `solid/SolidSolver.py`
```python
training_final_time = 1.  # Length of time period on which to train the ROM
rom_use = True
```

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

[4] A. Tiba, T. Dairay, F. Devuyst, I. Mortazavi, J-P. Berro Ramirez.
Non-intrusive reduced order models for partitioned fluid-structure interactions. arXiv:2306.07570, 2023.