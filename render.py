import h5py
import numpy as np
import scipy as sp
import scipy.sparse as sparse
import scipy.signal as signal
import scipy.ndimage
import matplotlib.cm as cm
import time


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


def compute_partitions(target_partitions, gbounds, grid_shape, overlap):
    xshape, yshape = grid_shape
    (gxmin, gxmax, gymin, gymax) = gbounds
    xlimit = gxmax - gxmin
    xslope = xlimit / (xshape - 1)
    chunks = []
    for idx in range(0, len(target_partitions) - 1):
        start = target_partitions[idx]
        end = target_partitions[idx + 1]
        start_idx = round((start - gxmin) / xslope)
        if start_idx < 0:
            start_idx = 0
        end_idx = round((end - gxmin) / xslope)
        if end_idx >= (xshape - 1):
            end_idx = (xshape - 1)
        start_idx_overlap = start_idx - overlap
        if start_idx_overlap < 0:
            start_idx_overlap = 0
        end_idx_overlap = end_idx + overlap
        if end_idx_overlap >= (xshape - 1):
            end_idx_overlap = (xshape - 1)
        start_val_overlap = gxmin + xslope * start_idx_overlap
        end_val_overlap = gxmin + xslope * end_idx_overlap
        start_val = gxmin + xslope * start_idx
        end_val = gxmin + xslope * end_idx
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
    xx, yy = np.ogrid[:2 * radius + 1,:2 * radius + 1]
    xx = xx - radius 
    yy = yy - radius 
    mask = np.sqrt(xx ** 2 +  yy **2) < radius
    return mask


import numba
import math

### project - takes 1d vectors, and maps those values and maps them on to
### a 2d grid (2d binning, similar to np.histogram2d
### exception is that we weight the values on neighboring pixels, instead of
### just having them fall into a bin
def project(xdata, ydata, grid, xmin, xmax, ymin, ymax, mark):
    xshape = grid.shape[0]
    yshape = grid.shape[1]
    max_x = xshape - 1
    max_y = yshape - 1
    xlimit = xmax - xmin
    ylimit = ymax - ymin
    xslope = xlimit / (xshape - 1)
    yslope = ylimit / (yshape - 1)
    mark_offset_x = int(math.floor(mark.shape[0] / 2))
    mark_offset_y = int(math.floor(mark.shape[1] / 2))
    for idx in range(len(xdata)):
        xcoord = (xdata[idx] - xmin) / xslope
        ycoord = (ydata[idx] - ymin) / yslope
        xcoord_int = int(math.floor(xcoord))
        ycoord_int = int(math.floor(ycoord))
        xrem = xcoord - xcoord_int
        yrem = ycoord - ycoord_int
        xbase = 1 - xrem
        ybase = 1 - yrem
        xoffset = xcoord_int + 1
        yoffset = ycoord_int + 1
        xx = xcoord_int
        yy = ycoord_int
        factor = xbase * ybase
        xx2 = xx - mark_offset_x
        yy2 = yy - mark_offset_y
        for c1 in range(mark.shape[0]):
            for c2 in range(mark.shape[1]):
                xx1 = xx2 + c1
                yy1 = yy2 + c2
                if xx1 > 0 and xx1 < xshape and xx1 > 0 and yy1 <yshape and yy1>0:
                    grid[xx1, yy1] += mark[c1, c2] * factor
        xx = xoffset
        yy = ycoord_int
        factor = xrem * ybase
        xx2 = xx - mark_offset_x
        yy2 = yy - mark_offset_y
        for c1 in range(mark.shape[0]):
            for c2 in range(mark.shape[1]):
                xx1 = xx2 + c1
                yy1 = yy2 + c2
                if xx1 > 0 and xx1 < xshape and xx1 > 0 and yy1 <yshape and yy1>0:
                    grid[xx1, yy1] += mark[c1, c2] * factor
        
        xx = xcoord_int
        yy = yoffset
        factor = xbase * yrem
        xx2 = xx - mark_offset_x
        yy2 = yy - mark_offset_y
        for c1 in range(mark.shape[0]):
            for c2 in range(mark.shape[1]):
                xx1 = xx2 + c1
                yy1 = yy2 + c2
                if xx1 > 0 and xx1 < xshape and xx1 > 0 and yy1 <yshape and yy1>0:
                    grid[xx1, yy1] += mark[c1, c2] * factor
        
        xx = xoffset
        yy = yoffset
        factor = xrem * yrem
        xx2 = xx - mark_offset_x
        yy2 = yy - mark_offset_y
        for c1 in range(mark.shape[0]):
            for c2 in range(mark.shape[1]):
                xx1 = xx2 + c1
                yy1 = yy2 + c2
                if xx1 > 0 and xx1 < xshape and xx1 > 0 and yy1 <yshape and yy1>0:
                    grid[xx1, yy1] += mark[c1, c2] * factor
    return grid
fast_project = numba.jit(project, nopython=True)
#fast_project = project
        

        # xx1 = xx - mark_offset_x
        # xx2 = yy + mark_offset_x
        # yy1 = yy - mark_offset_y
        # yy2 = yy + mark_offset_y
        # spill_xx1 = max(0 - xx1, 0)
        # spill_xx2 = max(xx2 - max_x, 0)
        # spill_yy1 = max(0 - yy1, 0)
        # spill_yy2 = max(yy2 - max_y, 0)
        # gx1 = xx - spill_xx1
        # gx2 = xx + spill_xx2
        # gy1 = yy - spill_yy1
        # gy2 = yy + spill_yy2
        
        # if xx < xshape and xx > 0 and yy <yshape and yy>0:
        #     grid[xx - spill_xx1 : xx + spill_xx2, yy - spill_yy1 : yy + spill_yy2] = mark[spill_xx1:spill_xx2, spill_yy1:spill_yy2] * xrem * ybase
        # xx = xcoord_int
        # yy = yoffset
        # xx1 = xx - mark_offset_x
        # xx2 = yy + mark_offset_x
        # yy1 = yy - mark_offset_y
        # yy2 = yy + mark_offset_y
        # spill_xx1 = max(0 - xx1, 0)
        # spill_xx2 = max(xx2 - max_x, 0)
        # spill_yy1 = max(0 - yy1, 0)
        # spill_yy2 = max(yy2 - max_y, 0)
        # gx1 = xx - spill_xx1
        # gx2 = xx + spill_xx2
        # gy1 = yy - spill_yy1
        # gy2 = yy + spill_yy2
        
        # if xx < xshape and xx > 0 and yy <yshape and yy>0:
        #     grid[xx - spill_xx1 : xx + spill_xx2, yy - spill_yy1 : yy + spill_yy2] = mark[spill_xx1:spill_xx2, spill_yy1:spill_yy2] * 
        # xx = xoffset
        # yy = yoffset
        # xx1 = xx - mark_offset_x
        # xx2 = yy + mark_offset_x
        # yy1 = yy - mark_offset_y
        # yy2 = yy + mark_offset_y
        # spill_xx1 = int(max(0 - xx1, 0))
        # spill_xx2 = int(max(xx2 - max_x, 0))
        # spill_yy1 = int(max(0 - yy1, 0))
        # spill_yy2 = int(max(yy2 - max_y, 0))
        # gx1 = xx - spill_xx1
        # gx2 = xx + spill_xx2
        # gy1 = yy - spill_yy1
        # gy2 = yy + spill_yy2
        # if xx < xshape and xx > 0 and yy <yshape and yy>0:
        #     grid[xx - spill_xx1 : xx + spill_xx2, yy - spill_yy1 : yy + spill_yy2] = mark[spill_xx1:spill_xx2, spill_yy1:spill_yy2] * xrem * yrem

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

def render_chunk_ks(chunked, boolean_partition, chunk_spec, xfield, yfield):
    pass
def render_chunked_ks(chunked_data, boolean_partitions,
                      chunk_specs,
                      grid_data_bounds, grid_shape, mark,
                      xfield, yfield
):
    (gxmin, gxmax, gymin, gymax) = grid_data_bounds
    for boolean_partition, chunk_spec in zip(boolean_partitions, chunk_specs):
        (overlap, start_val, end_val,
         start_val_overlap, end_val_overlap,
         start_idx, end_idx,
         start_idx_overlap, end_idx_overlap) = chunk_spec
        pass

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
