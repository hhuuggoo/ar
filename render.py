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
overlap = 3 #(in pixels)
target_partitions = np.array([-123.1139268, -100.3898881,  -99.1809685,  -89.62     ,
                              -75.6945583,  -74.61666  ,  -74.0758333,  -74.0758333,
                              -73.120468 ,  -58.4583333])

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
    xunit = (gxmax - gxmin) / gxgrid
    yunit = (gymax - gymin) / gygrid
    lxdim1 = (lxmin - gxmin) / xunit
    lxdim2 = (lxmax - gxmin) / xunit
    lydim1 = (lymin - gymin) / yunit
    lydim2 = (lymax - gymin) / yunit
    grid_shape = (gxgrid, gygrid)
    grid_data_bounds = (gxmin, gxmax, gymin, gymax)
    local_indexes = (round(lxdim1), round(lxdim2), round(lydim1), round(lydim2))

    units = (xunit, yunit)
    return grid_shape, grid_data_bounds, local_indexes, units

grid_shape, grid_data_bounds, local_indexes, units = discrete_bounds(
    (xscale, yscale),
    lbounds,
    gbounds,
    lxres, lyres)

def compute_partitions(target_partitions, gbounds, grid_shape, overlap):
    xshape, yshape = grid_shape
    (gxmin, gxmax, gymin, gymax) = gbounds
    xlimit = xmax - xmin
    xslope = xlimit / (xshape - 1)
    xunit, yunit = units
    chunks = []
    for idx in range(0, len(target_partitions) - 1):
        start = target_partitions[idx]
        end = target_partitions[idx + 1]
        start_idx = round((start - xmin) / xslope)
        end_idx = round((end - xmin) / xslope)
        start_idx_overlap = start_idx - overlap
        if start_idx_overlap < 0:
            start_idx_overlap = 0
        end_idx_overlap = end_idx + overlap
        if end_idx_overlap >= (xshape - 1):
            end_idx_overlap = (xshape - 1)
        start_val_overlap = xmin + xslope * start_idx_overlap
        end_val_overlap = xmin + xslope * end_idx_overlap
        start_val = xmin + xslope * start_idx
        end_val = xmin + xslope * end_idx
        chunk_info = (
            overlap,
            start_val, end_val,
            start_val_overlap, end_val_overlap,
            start_idx, end_idx,
            start_idx_overlap, end_idx_overlap)
        chunks.append(chunk_info)
    return chunks


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
def render(xdata, ydata, grid, grid_data_bounds, shape):
    output = fast_project(xdata, ydata,
                          grid,
                          *grid_data_bounds)
    output = scipy.ndimage.convolve(output, shape)
    return output

class XChunkedGrid(object):
    def __init__(self, data, yshape):
        """data is a list of x offsets, and a 2d array
        """
        self.data = data
        self.yshape = yshape

    def get(self, xstart, xend):
        bigdata = np.zeros(((xend - xstart), self.yshape))
        xsize = xend - xstart
        for x1, x2, data in self.data:
            xx1 = max(x1, xstart)
            xx2 = min(x2, xend)
            if xx2 - xx1 > 0:
                bigdata[xx1 - xstart:xx2 - xstart, :] = data[xx1 - x1:xx2 - x1, :]
        return bigdata


def render_chunked(xdata, ydata, chunks, grid_data_bounds, grid_shape, mark):
    (gxmin, gxmax, gymin, gymax) = gbounds
    data = []
    for chunk_info in chunks:
        (overlap, start_val, end_val,
         start_val_overlap, end_val_overlap,
         start_idx, end_idx,
         start_idx_overlap, end_idx_overlap) = chunk_info
        selector = [(xdata > start_val) & (xdata < end_val)]
        xchunk = xdata[selector]
        ychunk = ydata[selector]
        grid = np.zeros((end_idx_overlap - start_idx_overlap, grid_shape[1]))
        bounds = (start_val_overlap, end_val_overlap, gymin, gymax)
        output = fast_project(xchunk, ychunk, grid, *bounds)
        output = scipy.ndimage.convolve(output, mark)
        st = start_idx - start_idx_overlap
        ed = end_idx - start_idx_overlap
        output = output[st:ed, :]
        data.append((start_idx, end_idx, output))
    return XChunkedGrid(data, grid_shape[-1])

chunks = compute_partitions(target_partitions, gbounds, grid_shape, overlap)
chunked = render_chunked(xdata, ydata, chunks, grid_data_bounds, grid_shape, circle(2))
(lxdim1, lxdim2, lydim1, lydim2) = local_indexes
output = chunked.get(0, grid_shape[0] - 1)
output2 = chunked.get(lxdim1, lxdim2)[:, lydim1:lydim2]
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

#
# st = time.time()
# grid = np.zeros(grid_shape)
# output = render(xdata, ydata, grid, grid_data_bounds, circle(2))
# (lxdim1, lxdim2, lydim1, lydim2) = local_indexes
# output2 = output[lxdim1:lxdim2, lydim1:lydim2]
# ed = time.time()
# print ed-st

# ### after projecting individual points, convolve a shape(circle/square/cross)
# ### over the points
# import pylab
# pylab.imshow(output.T[::-1,:] ** 0.2,
#              cmap=cm.Greys_r,
#              extent=grid_data_bounds,
#              interpolation='nearest')
# pylab.figure()
# pylab.imshow(output2.T[::-1,:] ** 0.2,
#              cmap=cm.Greys_r,
#              extent=lbounds,
#              interpolation='nearest')
