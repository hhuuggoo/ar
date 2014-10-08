import h5py
import numpy as np
import scipy as sp
import scipy.signal as signal
import matplotlib.cm as cm
import time

f = h5py.File('latlong3.hdf5', mode='r')
ds = f['data']
ds = ds[::10000, 'latitude','longitude']
ds = ds[~(np.isnan(ds['longitude']) | np.isnan(ds['latitude']))]

def bounds(ds):
    longs = ds['longitude']
    xmin = longs.min()
    xmax = longs.max()

    lats = ds['latitude']
    ymin = lats.min()
    ymax = lats.max()
    return (xmin, xmax, ymin, ymax)

xmin, xmax, ymin, ymax = bounds(ds)
lxmin, lxmax, lymin, lymax = (-100.0, 100.0, -30., 30.)

xres = 600
yres = 400

xscale = (xmax-xmin) / (lxmax-lxmin)
yscale = (ymax-ymin) / (lymax-lymin)

gxres = (xres*xscale)
gyres = (yres*yscale)

gxgrid = np.ceil(gxres)
gygrid = np.ceil(gyres)

gxmin = xmin
gxmax = gxgrid / gxres * xmax
gymin = ymin
gymax = gygrid / gyres * ymax

global_image = np.zeros(gygrid, gxgrid)

def circle(radius=3):
    # todo: better circle
    xx, yy = np.ogrid[:2 * radius,:2 * radius]
    xx = xx - radius + 0.5
    yy = yy - radius + 0.5
    mask = np.sqrt(xx ** 2 +  yy **2) < radius
    return mask

import numba
def project(xdata, ydata, grid, xmin, xmax, ymin, ymax):
    xoffset = xmin
    xlimit = xmax - xmin
    ylimit = ymax - ymin
    xscale = xlimit / (grid.shape[0] - 1)
    yoffset = ymin
    yscale = ylimit / (grid.shape[1] - 1)
    xcoords = (xdata - xmin) / xscale
    ycoords = (ydata - ymin) / yscale
    xcoords_int = np.floor(xcoords)
    ycoords_int = np.floor(ycoords)
    xcoords_decimal = xcoords - xcoords_int
    ycoords_decimal = ycoords - ycoords_int

    for idx in range(len(xdata)):
        xbase = 1 - xcoords_decimal[idx]
        ybase = 1 - ycoords_decimal[idx]
        xrem = xcoords_decimal[idx]
        yrem = ycoords_decimal[idx]
        grid[xcoords_int[idx], ycoords_int[idx]] += xbase * ybase
        if xcoords_int[idx] + 1 < grid.shape[0]:
            grid[xcoords_int[idx] + 1, ycoords_int[idx]] += xrem * ybase
        if ycoords_int[idx] + 1 < grid.shape[1]:
            grid[xcoords_int[idx], ycoords_int[idx] + 1] += xbase * yrem
        if ycoords_int[idx] + 1 < grid.shape[1] and xcoords_int[idx] + 1 < grid.shape[0]:
            grid[xcoords_int[idx] + 1, ycoords_int[idx] + 1] += xrem * yrem
    points = grid
    return points

print 'start'

fast_project = numba.autojit(project)
st = time.time()
output = fast_project(ds['longitude'], ds['latitude'],
                      np.zeros((gxgrid, gygrid)),
                      gxmin, gxmax, gymin, gymax)
ed = time.time()
print ed-st
st = time.time()
output = project(ds['longitude'], ds['latitude'],
                 np.zeros((gxgrid, gygrid)),
                 gxmin, gxmax, gymin, gymax)
ed = time.time()
print ed-st
output = signal.convolve2d(output, circle())
output = output[:, ::-1]
import pylab
pylab.imshow(output ** 0.3, cmap=cm.Greys_r)
