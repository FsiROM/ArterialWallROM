import vtk
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

T = 1200  # number of timesteps performed

arraynames = ["diameter","pressure","velocity", "iters"]   # Which dataset should be plotted?
data_path = "../../fluid-python/output/out_fluid_"  # Where is the data?


def file_name_generator(id): return data_path + str(id) + ".vtk"


print("parsing datasets named %s*.vtk" % data_path)

values_for_all_t = T * [None]

dx = .1
dt = .1
N = 100
mesh = np.arange(0, dx*(N+1), dx)
time = np.arange(0, dt*T, dt)+dt

data_ = []
for name in arraynames:
    for t in range(T):

        # read the vtk file as an unstructured grid
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(file_name_generator(t))
        reader.ReadAllVectorsOn()
        reader.ReadAllScalarsOn()
        reader.Update()

        # parse the data
        grid = reader.GetOutput()
        point_data = grid.GetPointData().GetArray(name)
        points = grid.GetPoints()
        N = grid.GetNumberOfPoints()  # How many gridpoints do exist?

        if point_data is None:  # check if array exists in dataset
            print("array with name %s does not exist!" % name)
            print("exiting.")
            quit()

        value_at_t = []
        spatial_mesh = []

        n = point_data.GetNumberOfComponents()

        for i in range(N):  # parse data from vtk array into list

            x, y, z = grid.GetPoint(i)  # read coordinates of point
            spatial_mesh += [x]  # only store x component

            v = np.zeros(n)  # initialize empty butter array
            point_data.GetTuple(i, v)  # read value into v
            value_at_t += [v[0]]

        values_for_all_t[t] = value_at_t


    values_for_all_t = np.array(values_for_all_t)
    
    data_.append(values_for_all_t.copy().T)

data_[3] = data_[3][0, :].copy()

np.save("pressure.npy", data_[1])
np.save("iters.npy", data_[3])
np.save("diameter.npy", data_[0])
np.save("velocity.npy", data_[2])

