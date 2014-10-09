import os
import math
from os.path import relpath, join, basename, exists
import time

import numpy as np
import h5py
import pandas as pd
import scipy.ndimage
import cStringIO as StringIO
from kitchensink import setup_client, client, do, du, dp

from search import Chunked, smartslice
from render import (compute_partitions, discrete_bounds, compute_scales, circle)
setup_client('http://power:6323/')


def clean_lat(x):
    return (x != 0) & ~np.isnan(x) & (x >= 40.5) & (x <= 41.)

def clean_long(x):
    return (x != 0) & ~np.isnan(x) & (x >= -74.2) & (x <= -73.7032)

class ARDataset(object):
    overlap = 3
    lxres = 800
    lyres = 400
    target_partitions = np.array([-74.191536, -74.001808, -73.994064,
                                  -73.989662, -73.985313,
                                  -73.981331, -73.976891, -73.971153,
                                  -73.962082, -73.7 ])
    scales = np.array([1, 2, 4, 8, 16]).astype('float64')
    gbounds = (-74.19, -73.7, 40.5, 40.99)
    cache = {}

    def __init__(self, local_bounds):
        self.lbounds = local_bounds
        xscale, yscale = compute_scales(self.lbounds, self.gbounds,
                                        self.scales)
        self.xscale, self.yscale = xscale, yscale
        grid_shape, grid_data_bounds, local_indexes, units = discrete_bounds(
            (xscale, yscale),
            self.lbounds,
            self.gbounds,
            self.lxres,
            self.lyres
        )
        self.grid_shape = grid_shape
        self.grid_data_bounds = grid_data_bounds
        self.local_indexes = local_indexes
        self._cleaned = None
        self._chunked = None
        self._partitions = None
        self.mark = circle()
    def partitions(self):
        c = client()
        url = 'taxi/partitioned/sources'
        if self._partitions:
            return self._partitions
        if c.path_search(url):
            self._partitions = du(url).obj()
            return self._partitions

    def partition_indices(self):
        c = client()
        # we use zoom1 grid shape to compute partitions
        # that way partitions don't need to change
        # as we zoom - it's just that we might have extra overlap
        partition_grid_shape = [self.lxres, self.lyres]
        url = 'taxi/indexes/partitioned'
        if self._partition_indices:
            return self._partition_indices
        if c.path_search(url):
            self._partition_indices = du(url).obj()
            return self._partition_indices
        chunked = self.chunked()
        partitions = compute_partitions(self.target_partitions,
                                        self.gbounds,
                                        grid_shape,
                                        self.overlap)
        partitioned_data = []
        for p in partitions:
            start_val_overlap, end_val_overlap = p[3:5]
            print start_val_overlap, end_val_overlap
            st = start_val_overlap
            ed = end_val_overlap
            def helper(data):
                val = (data >= st) & (data <= ed)
                return val
            results = chunked.query(
                {'pickup_longitude' : [helper]},
                prefilter=self.cleaned_data()
            )
            print [x.obj().sum() for x in results]
            partitioned_data.append(results)
        self._partition_indices = zip(partitions, partitioned_data)
        do(self._partition_indices).save(url=url)
        return self._partition_indices

    def chunked(self):
        if self._chunked:
            return self._chunked
        c = client()
        urls = c.path_search('taxi/*hdf5')
        urls.sort()
        objs = [du(x) for x in urls]
        chunked = Chunked(objs)
        #compute the property, for kicks
        chunked.chunks
        self._chunked = chunked
        return self._chunked

    def cleaned_data(self):
        if self._cleaned:
            return self._cleaned
        c = client()
        if c.path_search('taxi/cleaned'):
            self._cleaned = du('taxi/cleaned').obj()
            return self._cleaned
        chunked = self.chunked()
        cleaned = chunked.query({'pickup_latitude' : [clean_lat],
                                 'pickup_longitude' : [clean_long],
                                 'dropoff_latitude' : [clean_lat],
                                 'dropoff_longitude' : [clean_long],
                             })
        self._cleaned = cleaned
        do(self._cleaned).save(url='taxi/cleaned')
        return self._cleaned

    def project(self):
        c = client()
        for partition_spec, partitioned_data in self.partitions():
            c.bc(render, self.chunked().chunks, partition_spec, partitioned_data,
                 self.gbounds, self.grid_shape, self.mark,
                 'pickup_longitude', 'pickup_latitude')
            # c.execute()
            # c.br()
        c.execute()
        results = c.br()
        return results

def render(chunks, partition_spec, boolean_objs,
           grid_data_bounds, grid_shape, mark, xfield, yfield,
       ):
    from render import fast_project
    (overlap, start_val, end_val,
     start_val_overlap, end_val_overlap,
     start_idx, end_idx,
     start_idx_overlap, end_idx_overlap) = partition_spec
    gxmin, gxmax, gymin, gymax = grid_data_bounds
    xdatas = []
    ydatas = []
    grid = np.zeros((end_idx_overlap - start_idx_overlap, grid_shape[1]))
    bounds = (start_val_overlap, end_val_overlap, gymin, gymax)
    for boolean_obj, (source, start, end) in zip(boolean_objs, chunks):
        bvector = boolean_obj.obj()
        path = source.local_path()
        f = h5py.File(path, 'r')
        try:
            ds = f[xfield]
            xdata = smartslice(ds, start, end, bvector)
            ds = f[yfield]
            ydata = smartslice(ds, start, end, bvector)
        finally:
            f.close()
        print xdata.shape, ydata.shape, path
        fast_project(xdata, ydata, grid, *bounds)
    scipy.ndimage.convolve(grid, mark)
    print np.max(grid)
    st = start_idx - start_idx_overlap
    ed = end_idx - start_idx_overlap
    grid = grid[st:ed, :]
    obj = do(grid)
    obj.save(prefix="projection")
    return start_idx, end_idx, obj

class KSXChunkedGrid(object):
    def __init__(self, data, yshape):
        """data is a list of x offsets, and a 2d array
        """
        self.data = data
        self.yshape = yshape

    def get(self, xstart, xend):
        c = client()
        bigdata = np.zeros(((xend - xstart), self.yshape))
        xsize = xend - xstart
        assignments = []
        def helper(data, x1, xx1, xx2):
            data = data.obj()
            return data[xx1-x1:xx2-x1, :]

        for x1, x2, data in self.data:
            xx1 = max(x1, xstart)
            xx2 = min(x2, xend)
            if xx2 - xx1 > 0:
                assignments.append((xx1 - xstart, xx2 - xstart))
                c.bc(helper, data, x1, xx1, xx2)
        c.execute()
        results = c.br()
        for assignment, output in zip(assignments, results):
            bigdata[assignment[0]:assignment[1], :] = output
        return bigdata


if __name__ == "__main__":
    #client().reducetree('taxi/cleaned*')
    #client().reducetree('taxi/index*')
    import matplotlib.cm as cm
    st = time.time()
    ds = ARDataset((-74.19, -73.7, 40.5, 40.999))
    result = ds.partitions()
    result = ds.project()
    grid = KSXChunkedGrid(result, ds.grid_shape[-1])
    output = grid.get(0, ds.grid_shape[0] - 1)
    lxdim1, lxdim2, lydim1, lydim2 = ds.local_indexes
    output2 = grid.get(lxdim1, lxdim2)[:, lydim1:lydim2]
    ed = time.time()
    import pylab
    pylab.imshow(output.T[::-1,:] ** 0.3,
                 cmap=cm.Greys_r,
                 #extent=ds.grid_data_bounds,
                 interpolation='nearest')
    pylab.figure()
    pylab.imshow(output2.T[::-1,:] ** 0.3,
                 cmap=cm.Greys_r,
                 #extent=ds.lbounds,
                 interpolation='nearest')
    pylab.show()
