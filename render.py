import h5py
import numpy as np
import scipy as sp
import scipy.sparse as sparse
import scipy.signal as signal
import scipy.ndimage
import matplotlib.cm as cm
import time

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

import math

if __name__ == "__main__":
    ### grab 2 1d vectors (lat/long) from hdf5 file
    shape = circle()
    f = h5py.File('columnar.hdf5', mode='r')
    ydata = f['latitude'][:6000000]
    xdata = f['longitude'][:6000000]
    selector = ~(np.isnan(xdata) | np.isnan(xdata) | (xdata == 0) | (xdata == 0))

    xdata = xdata[selector]
    ydata = ydata[selector]

    ##constants
    (xmin, xmax, ymin, ymax) = (-123.11392679999999,
                                153.4532255,
                                -54.801912100000003,
                                64.9833)
    scales = np.array([1, 2, 4, 8, 16]).astype('float64')
    lxmin, lxmax, lymin, lymax = (-100., -70., -10., 0.)
    lxres = 600
    lyres = 400
    overlap = 3 #(in pixels)
    target_partitions = np.array([-123.1139268, -100.3898881,  -99.1809685,  -89.62     ,
                                  -75.6945583,  -74.61666  ,  -74.0758333,  -74.0758333,
                                  -73.120468 ,  -58.4583333])

    lbounds = (lxmin, lxmax, lymin, lymax)
    gbounds = (xmin, xmax, ymin, ymax)
    xscale, yscale = compute_scales(lbounds, gbounds, scales)

    grid_shape, grid_data_bounds, local_indexes, units = discrete_bounds(
        (xscale, yscale),
        lbounds,
        gbounds,
        lxres, lyres)


    chunks = compute_partitions(target_partitions, gbounds, grid_shape, overlap)
    chunked = render_chunked(xdata, ydata, chunks, grid_data_bounds, grid_shape, circle(2))
    (lxdim1, lxdim2, lydim1, lydim2) = local_indexes
    output = chunked.get(0, grid_shape[0] - 1)
    output2 = chunked.get(lxdim1, lxdim2)[:, lydim1:lydim2]

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
