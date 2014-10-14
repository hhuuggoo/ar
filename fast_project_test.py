import pandas as pd
import time
from fast_project import project
import numpy as np
import h5py
path = "/home/hugo/ramdisk/big.hdf5"
f = h5py.File(path)
ydata = f['pickup_latitude'][:1000000]
xdata = f['pickup_longitude'][:1000000]
print xdata, ydata
xmin, xmax, ymin, ymax = (-74.05, -73.75, 40.5, 40.99)
marker = np.array([[1]]).astype('float64') #unused, but signature still has it

grid = np.zeros((300,300))
st = time.time()
project(xdata, ydata, grid, xmin, xmax, ymin, ymax, marker)
ed = time.time()
print ed-st
print np.max(grid)
