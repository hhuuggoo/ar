import os
import math
from os.path import relpath, join, basename, exists, dirname
import time

import numpy as np
import tempfile
import h5py
import pandas as pd
import scipy.ndimage
import cStringIO as StringIO
from kitchensink import setup_client, client, do, du, dp
from kitchensink import settings
from search import Chunked, smartslice, boolfilter
from render import (compute_partitions, discrete_bounds, compute_scales, circle)
setup_client('http://power:6323/')


def clean_lat(x):
    return (x != 0) & ~np.isnan(x) & (x >= 40.5) & (x <= 41.)

def clean_long(x):
    return (x != 0) & ~np.isnan(x) & (x >= -74.2) & (x <= -73.7032)

def _h5py_append(src, dst, start, end, boolvect):
    print src, dst
    if not exists(dirname(dst)):
        os.makedirs(dirname(dst))
    src_file = h5py.File(src, 'r')
    dst_file = h5py.File(dst, 'a')
    try:
        for k in src_file.keys():
            print (dst, k)
            data = smartslice(src_file[k], start, end, boolvect)
            if len(data) == 0:
                continue
            if k not in dst_file.keys():
                dst_file.create_dataset(
                    k, data=data, maxshape=(None,), compression='lzf'
                )
            else:
                ds = dst_file[k]
                orig = ds.shape[0]
                ds.resize((orig + len(data),))
                ds[orig:] = data
    finally:
        src_file.close()
        dst_file.close()

def h5py_append(dst_path, chunks, boolobjs):
    for boolobj, (source, start, end) in zip(boolobjs, chunks):
        src_path = source.local_path()
        _h5py_append(src_path, dst_path, start, end, boolobj.obj())

class ARDataset(object):
    overlap = 3
    lxres = 800
    lyres = 400
    target_partitions = np.array([-74.191536, -74.001808, -73.994064,
                                  -73.981331, -73.976891, -73.971153,
                                  -73.962082, -73.7 ])
    scales = np.array([1, 2, 4, 8, 16]).astype('float64')
    gbounds = (-74.19, -73.7, 40.5, 40.99)
    cache = {}

    def __init__(self):
        self._cleaned = None
        self._chunked = None
        self._partitions = None
        self._partition_indices = None
        self.mark = circle(radius=1.0)

    def partitions(self):
        c = client()
        url = 'taxi/partitioned/sources'
        if self._partitions:
            return self._partitions
        if c.path_search(url):
            self._partitions = du(url).obj()
            return self._partitions
        chunked = self.chunked()
        paths = []
        for idx, (partition_spec, partitioned_data) in \
            enumerate(self.partition_indices()):

            boolean_objs = partitioned_data
            chunks = chunked.chunks
            #HACK
            dst_path = join("/data", 'taxi', 'partitioned', str(idx))
            paths.append(join('taxi', 'partitioned', str(idx)))
            c.bc(h5py_append, dst_path, chunks, boolean_objs)
        c.execute()
        c.br()
        for p in paths:
            c.bc('bootstrap', p,
                 data_type='file', _queue_name='data|power', _rpc_name='data')
        c.execute()
        c.br()
        obj = do([du(x) for x in paths])
        obj.save(url=url)
        self._partitions = obj.obj()
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
                                        partition_grid_shape,
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

    def query(self, query_dict):
        c = client()
        chunked = Chunked(self.partitions())
        for source, start, end in chunked.chunks:
            c.bc(boolfilter, source, start, end, query_dict)
        c.execute()
        results = c.br()
        output = {}
        for result, (source, start, end) in zip(results, chunked.chunks):
            output[(source.data_url, start, end)] = result
        output = do(output)
        output.save(prefix='taxi/query')
        return output

    def project(self, local_bounds, xfield, yfield, filters=None):
        c = client()
        xscale, yscale = compute_scales(local_bounds, self.gbounds,
                                        self.scales)
        grid_shape, grid_data_bounds, local_indexes, units = discrete_bounds(
            (xscale, yscale),
            local_bounds,
            self.gbounds,
            self.lxres,
            self.lyres
        )
        if filters:
            t = (filters.data_url, xfield, yfield, grid_shape[0], grid_shape[1])
            url = "taxi/projections/%s/%s/%s/%s/%s" % t
        else:
            t = (xfield, yfield, grid_shape[0], grid_shape[1])
            url = "taxi/projections/%s/%s/%s/%s" % t
        if filters is not None:
            filters = filters.obj()
        else:
            filters = {}
        if url in self.cache:
            return local_indexes, self.cache[url]
        if c.path_search(url):
            return local_indexes, du(url).obj()
        c = client()
        # these should matche the data partitions, except
        # the actual overlap may be reduced due to different
        # zoom level
        partition_specs = compute_partitions(self.target_partitions,
                                             self.gbounds,
                                             grid_shape,
                                             self.overlap)
        print 'PROJECTIONG'
        for partition_info, partition in zip(partition_specs, self.partitions()):
            chunked = Chunked([partition])
            c.bc(render, chunked.chunks, partition_info, filters, self.gbounds,
                 grid_shape, self.mark,
                 'pickup_longitude', 'pickup_latitude')
        c.execute()
        results = c.br()
        print 'DONE PROJECTIONG'
        self.cache[url] = grid_shape, results
        do(self.cache[url]).save(url)
        return local_indexes, self.cache[url]

def render(chunks, partition_spec, filters,
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
    for source, start, end in chunks:
        boolean_obj = filters.get((source.data_url, start, end))
        if boolean_obj is not None:
            bvector = boolean_obj.obj()
        else:
            bvector = None
        print source.data_url, start, end, bvector
        path = source.local_path()
        f = h5py.File(path, 'r')
        try:
            ds = f[xfield]
            xdata = smartslice(ds, start, end, bvector)
            ds = f[yfield]
            ydata = smartslice(ds, start, end, bvector)
        finally:
            f.close()
        print xdata.shape, ydata.shape, grid.shape, path, bounds
        fast_project(xdata, ydata, grid, *bounds)
    grid = scipy.ndimage.convolve(grid, mark)
    st = start_idx - start_idx_overlap
    ed = end_idx - start_idx_overlap
    grid = grid[st:ed, :]
    print np.max(grid)
    path = tempfile.NamedTemporaryFile().name
    f = h5py.File(path)
    f.create_dataset('data', data=grid, compression='lzf')
    f.close()
    obj = dp(path)
    obj.save(prefix="projection")
    return start_idx, end_idx, obj

class KSXChunkedGrid(object):
    def __init__(self, data, yshape):
        """data is a list of x offsets, and a 2d array
        """
        self.data = data
        self.yshape = yshape

    def get(self, xstart, xend, ystart, yend):
        c = client()
        bigdata = np.zeros((int(xend - xstart), int(yend-ystart)))
        xsize = xend - xstart
        assignments = []
        def helper(data, x1, xx1, xx2):
            data = h5py.File(data.local_path())['data']
            return data[int(xx1-x1):int(xx2-x1), int(ystart):int(yend)]
        for x1, x2, data in self.data:
            xx1 = max(x1, xstart)
            xx2 = min(x2, xend)
            if xx2 - xx1 > 0:
                assignments.append((xx1 - xstart, xx2 - xstart))
                c.bc(helper, data, x1, xx1, xx2)
        c.execute()
        results = c.br()
        print assignments
        for assignment, output in zip(assignments, results):
            bigdata[assignment[0]:assignment[1], :] = output
        return bigdata


if __name__ == "__main__":
    #client().reducetree('taxi/partitioned*')
    #client().reducetree('taxi/cleaned*')
    #client().reducetree('taxi/index*')
    client().reducetree('taxi/projections*')
    import matplotlib.cm as cm
    st = time.time()
    ds = ARDataset()
    #filters = ds.query({'trip_time_in_secs' : [lambda x : (x >= 1999) & (x <= 2000)]})
    filters = None
    global_bounds = (-74.19, -73.7, 40.5, 40.999)
    local_bounds = (-74.0, -73.9, 40.7, 40.8)
    #local_bounds = global_bounds
    local_indexes, (grid_shape, results) = ds.project(local_bounds, filters)
    lxdim1, lxdim2, lydim1, lydim2 = local_indexes
    ed = time.time()
    print ed-st
    import pylab
    grid = KSXChunkedGrid(results, grid_shape[-1])
    # output = grid.get(0, grid_shape[0] - 1)
    # pylab.imshow(output.T[::-1,:] ** 0.3,
    #              cmap=cm.Greys_r,
    #              extent=global_bounds,
    #              interpolation='nearest')
    # pylab.figure()
    st = time.time()
    output2 = grid.get(lxdim1, lxdim2, lydim1, lydim2)
    ed = time.time()
    print ed - st
    pylab.imshow(output2.T[::-1,:] ** 0.3,
                 cmap=cm.Greys_r,
                 extent=local_bounds,
                 interpolation='nearest')
    pylab.show()
