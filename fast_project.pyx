import numpy as np
cimport numpy as np
ctypedef np.float64_t FLOAT_t
from libc.math cimport floor, ceil
import math
import cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.overflowcheck(False)
@cython.cdivision(False)
def project(np.ndarray[FLOAT_t, ndim=1] xdata, np.ndarray[FLOAT_t, ndim=1] ydata, np.ndarray[FLOAT_t, ndim=2] grid, float xmin, float xmax, float ymin, float ymax, np.ndarray[FLOAT_t, ndim=2] mark):
    cdef int xshape = grid.shape[0]
    cdef int yshape = grid.shape[1]
    cdef int max_x = xshape - 1
    cdef int max_y = yshape - 1
    cdef float xlimit = xmax - xmin
    cdef float ylimit = ymax - ymin
    cdef float xslope = xlimit / (xshape - 1)
    cdef float yslope = ylimit / (yshape - 1)
    cdef int mark_offset_x = <int> floor(mark.shape[0] / 2)
    cdef int mark_offset_y = <int> floor(mark.shape[1] / 2)
    cdef int idx
    cdef float xcoord, ycoord, yrem, xrem , xbase, ybase, factor
    cdef int xcoord_int, ycoord_int, xoffset, yoffset, xx, yy, xx2, yy2, xx1, yy1, c2, c1, xmark_shape, ymark_shape, datalen
    xmark_shape = mark.shape[0]
    ymark_shape = mark.shape[1]
    datalen = len(xdata)
    for idx in range(datalen):
        if xdata[idx] <= xmin or xdata[idx] >=xmax or \
           ydata[idx] <=ymin or ydata[idx]>=ymax:
            continue
        xcoord = (xdata[idx] - xmin) / xslope
        ycoord = (ydata[idx] - ymin) / yslope
        xcoord_int = (<int> floor(xcoord))
        ycoord_int = (<int> floor(ycoord))
        # xrem = xcoord - xcoord_int
        # yrem = ycoord - ycoord_int
        # xbase = 1 - xrem
        # ybase = 1 - yrem
        # xoffset = xcoord_int + 1
        # yoffset = ycoord_int + 1

        xx = xcoord_int
        yy = ycoord_int
        if xx > 0 and xx < xshape and xx > 0 and yy <yshape and yy>0:
            grid[xx, yy] += 1;
        # xx = xoffset
        # yy = ycoord_int
        # factor = xrem * ybase
        # if xx > 0 and xx < xshape and xx > 0 and yy <yshape and yy>0:
        #     grid[xx, yy] += factor
        # xx = xcoord_int
        # yy = yoffset
        # factor = xbase * yrem
        # if xx > 0 and xx < xshape and xx > 0 and yy <yshape and yy>0:
        #     grid[xx, yy] += factor
        # xx = xoffset
        # yy = yoffset
        # factor = xrem * yrem
        # if xx > 0 and xx < xshape and xx > 0 and yy <yshape and yy>0:
        #     grid[xx, yy] += factor
    return grid
