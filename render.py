import h5py
import numpy as np
import scipy as sp
import scipy.sparse as sparse
import scipy.signal as signal
import scipy.ndimage
import matplotlib.cm as cm
import time

### grab 2 1d vectors (lat/long) from hdf5 file

f = h5py.File('columnar.hdf5', mode='r')
st = time.time()
ydata = f['latitude'][:6000000]
xdata = f['longitude'][:6000000]
ed = time.time()
selector = ~(np.isnan(xdata) | np.isnan(xdata) | (xdata == 0) | (xdata == 0))
print ed - st
### remove nans/zeros

xdata = xdata[selector]
ydata = ydata[selector]


### some dumb code that computes data bounds
### and the size of the global canvas, assuming
### we are zoomed in to lxmin/lxmax lymin/lymax

def bounds(xdata, ydata):
    longs = xdata
    xmin = longs.min()
    xmax = longs.max()

    lats = ydata
    ymin = lats.min()
    ymax = lats.max()
    return (xmin, xmax, ymin, ymax)

##constants
xmin, xmax, ymin, ymax = bounds(xdata, ydata)
scales = np.array([1, 2, 4, 8, 16]).astype('float64')
lxmin, lxmax, lymin, lymax = (-123., -70., -10., 50.)
#lxmin, lxmax, lymin, lymax = xmin, xmax, ymin, ymax
lxres = 600
lyres = 400


def nearest_scale(target, all_scales):
    if target > all_scales[-1]:
        return all_scales[-1]
    else:
        return all_scales[(all_scales - target) >= 0][0]
## scales are zoom factor - scale = 1.0, means render 1x magnification
def compute_scales(lbounds, gbounds, all_scales):
    lxmin, lxmax, lymin, lymax = lbounds
    xmin, xmax, ymin, ymax = gbounds
    xscale = (xmax-xmin) / (lxmax-lxmin)
    yscale = (ymax-ymin) / (lymax-lymin)
    nearest_xscale = nearest_scale(xscale, all_scales)
    nearest_yscale = nearest_scale(yscale, all_scales)
    return (nearest_xscale, nearest_yscale)

lbounds = (lxmin, lxmax, lymin, lymax)
gbounds = (xmin, xmax, ymin, ymax)
xscale, yscale = compute_scales(lbounds, gbounds, scales)

def discrete_bounds(scales, lbounds, gbounds, lxres, lyres):
    xscale, yscale = scales
    lxmin, lxmax, lymin, lymax = lbounds
    xmin, xmax, ymin, ymax = gbounds
    #round up to the nearest pixel
    gxres = lxres*xscale
    gyres = lyres*yscale
    gxgrid = np.ceil(gxres)
    gygrid = np.ceil(gyres)
    gxmax = (gxgrid / gxres) * xmax
    gymax = (gygrid / gyres) * ymax
    (gxmin, gymin) = (xmin, ymin)

    #local offsets
    lxdim1 = (lxmin - gxmin) * gxgrid / (gxmax - gxmin)
    lxdim2 = (lxmax - gxmin) * gxgrid / (gxmax - gxmin)
    lydim1 = (lymin - gymin) * gygrid / (gymax - gymin)
    lydim2 = (lymax - gymin) * gygrid / (gymax - gymin)
    grid_shape = (gxgrid, gygrid)
    grid_data_bounds = (gxmin, gxmax, gymin, gymax)
    local_indexes = (lxdim1, lxdim2, lydim1, lydim2)
    return grid_shape, grid_data_bounds, local_indexes

grid_shape, grid_data_bounds, local_indexes = discrete_bounds(
    (xscale, yscale),
    lbounds,
    gbounds,
    lxres, lyres)

### dumb function that returns a circular-ish kernel for convolution
def circle(radius=3):
    # todo: better circle
    xx, yy = np.ogrid[:2 * radius,:2 * radius]
    xx = xx - radius + 0.5
    yy = yy - radius + 0.5
    mask = np.sqrt(xx ** 2 +  yy **2) < radius
    return mask

shape = circle()

import numba
import math

### project - takes 1d vectors, and maps those values and maps them on to
### a 2d grid (2d binning, similar to np.histogram2d
### exception is that we weight the values on neighboring pixels, instead of
### just having them fall into a bin
def project(xdata, ydata, grid, xmin, xmax, ymin, ymax):
    xshape = grid.shape[0]
    yshape = grid.shape[1]
    xlimit = xmax - xmin
    ylimit = ymax - ymin
    xslope = xlimit / (xshape - 1)
    yslope = ylimit / (yshape - 1)
    for idx in range(len(xdata)):
        xcoord = (xdata[idx] - xmin) / xslope
        ycoord = (ydata[idx] - ymin) / yslope
        xcoord_int = int(math.floor(xcoord))
        ycoord_int = int(math.floor(ycoord))
        xrem = xcoord - xcoord_int
        yrem = ycoord - ycoord_int
        xbase = 1 - xrem
        ybase = 1 - yrem
        grid[xcoord_int, ycoord_int] += xbase * ybase
        xoffset = xcoord_int + 1
        yoffset = ycoord_int + 1
        if xoffset < xshape:
            grid[xoffset, ycoord_int] += xrem * ybase
        if yoffset < yshape:
            grid[xcoord_int, yoffset] += xbase * yrem
        if yoffset < yshape and xoffset < xshape:
            grid[xoffset, yoffset] += xrem * yrem
    return grid

fast_project = numba.jit(project, nopython=True)
st = time.time()
output = fast_project(xdata, ydata,
                      np.zeros(grid_shape),
                      *grid_data_bounds)
points = output
output = scipy.ndimage.convolve(points, circle(radius=2))
(lxdim1, lxdim2, lydim1, lydim2) = local_indexes
output2 = output[lxdim1:lxdim2, lydim1:lydim2]
ed = time.time()
print ed-st

### after projecting individual points, convolve a shape(circle/square/cross)
### over the points
import pylab
pylab.imshow(output.T[::-1,:] ** 0.2,
             cmap=cm.Greys_r,
             extent=grid_data_bounds,
             interpolation='nearest')
pylab.figure()
pylab.imshow(output2.T[::-1,:] ** 0.2,
             cmap=cm.Greys_r,
             extent=lbounds,
             interpolation='nearest')
