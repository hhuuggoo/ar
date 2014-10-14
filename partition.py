import os
import math
from os.path import relpath, join, basename, exists, dirname
import time
import datetime as dt
import numpy as np
import tempfile
import h5py
import pandas as pd
import scipy.ndimage
import cStringIO as StringIO
from kitchensink import setup_client, client, do, du, dp
from kitchensink import settings
from search import Chunked, smartslice, boolfilter

class ARDataset(object):
    overlap = 3
    lxres = 350.0
    lyres = 550.0
    gbounds = (-74.05, -73.75, 40.5, 40.99)
    scales = np.array([1, 2, 4, 8, 16, 32, 64, 128]).astype('float64')
    cache = {}

    def __init__(self):
        self._cleaned = None
        self._chunked = None
        self._partitions = None
        self._partition_indices = None
        self.mark = np.array([[1]])

    def clean_lat(self, x):
        return (x >= self.gbounds[2]) & (x <= self.gbounds[3])

    def clean_long(self, x):
        return (x >= self.gbounds[0]) & (x <= self.gbounds[1])

    def chunked(self):
        if self._chunked:
            return self._chunked
        c = client()
        urls = c.path_search('taxi/big.hdf5')
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
        cleaned = chunked.query({'pickup_latitude' : [self.clean_lat],
                                 'pickup_longitude' : [self.clean_long],
                                 'dropoff_latitude' : [self.clean_lat],
                                 'dropoff_longitude' : [self.clean_long],
                             })
        self._cleaned = cleaned
        do(self._cleaned).save(url='taxi/cleaned')
        return self._cleaned

    def project(self, local_bounds, xfield, yfield, filters=None):
        if filters is None:
            filters = {}
        else:
            filters = filters.obj()
        mark = self.mark
        grid_shape = [self.lxres, self.lyres]
        c = client()
        for source, start, end in self.chunked().chunks:
            c.bc(render, source, start, end, filters,
                 local_bounds, grid_shape, mark,
                 xfield, yfield, _intermediate_results=False,
                 _no_route_data=True)
        c.execute()
        results = c.br(profile='project_profile_%s' % xfield)
        return sum(results)

    def query(self, query_dict):
        c = client()
        chunked = self.chunked()
        for source, start, end in chunked.chunks:
            c.bc(boolfilter, source, start, end, query_dict, _intermediate_results=False, _no_route_data=True)
        c.execute()
        results = c.br(profile='profile_query')
        output = {}
        for result, (source, start, end) in zip(results, chunked.chunks):
            output[(source.data_url, start, end)] = result
        output = do(output)
        output.save(prefix='taxi/query')
        return output


    def aggregate(self, results, grid_shape):
        c = client()
        data_urls = [x.data_url for x in results]
        hosts, info = c.data_info(data_urls)
        process_dict = {}
        for u in data_urls:
            hosts, meta = info[u]
            assert len(hosts) == 1
            process_dict.setdefault(list(hosts)[0], []).append(u)
        c = client()
        for k, v in process_dict.items():
            v = [du(x) for x in v]
            queue_name = c.queue('default', host=k)
            c.bc(aggregate, v, grid_shape, _intermediate_results=False, _queue_name=queue_name)
        c.execute()
        results = c.br(profile='aggregate')
        results = [x.obj() for x in results]
        results = sum(results)
        return results

    def histogram(self, field, bins, filters=None):
        st = time.time()
        c = client()
        if filters is None:
            filters = {}
        else:
            filters = filters.obj()
        for source, start, end in self.chunked().chunks:
            c.bc(histogram, source, start, end, filters, field, bins, _intermediate_results=True, _no_route_data=True)
        ed = time.time()
        c.execute()
        return c

    def finish_histogram(self, results):
        counts = [x[0] for x in results]
        return np.array(counts).sum(axis=0)

from kitchensink.admin import timethis

def histogram(source, start, end, filters, field, bins):
    with timethis('loading'):
        boolean_obj = filters.get((source.data_url, start, end))
        if boolean_obj is not None:
            bvector = boolean_obj.obj()
        else:
            bvector = None
        path = source.local_path()
        f = h5py.File(path, 'r')
        try:
            ds = f[field]
            data = smartslice(ds, start, end, bvector)

        finally:
            f.close()
    with timethis('histogram'):
        result = np.histogram(data, bins)
    return result

def aggregate(results, grid_shape):
    with timethis('data_loading'):
        bigdata = np.zeros(grid_shape)
        for source in results:
            path = source.local_path()
            data = h5py.File(path)['data']
            bigdata += data[:,:]
    with timethis('saving_result'):
        obj = do(bigdata)
        obj.save(prefix='taxi/aggregate')
    return obj

from fast_project import project as fast_project
from kitchensink.admin import timethis
def render(source, start, end, filters, grid_data_bounds,
           grid_shape, mark, xfield, yfield):
    with timethis('init'):
        gxmin, gxmax, gymin, gymax = grid_data_bounds
        grid = np.zeros(grid_shape)
        boolean_obj = filters.get((source.data_url, start, end))
        if boolean_obj is not None:
            bvector = boolean_obj.obj()
        else:
            bvector = None
        path = source.local_path()
    with timethis('loading'):
        f = h5py.File(path, 'r')
        try:
            ds1 = f[xfield]
            xdata = smartslice(ds1, start, end, bvector)
            ds2 = f[yfield]
            ydata = smartslice(ds2, start, end, bvector)
        finally:
            f.close()
    with timethis('project'):
        mark = mark.astype('float64')
        args = (xdata, ydata, grid) + grid_data_bounds + (mark,)
        fast_project(*args)
    return grid


if __name__ == "__main__":
    setup_client('http://power:6323/')
    #client().reducetree('taxi/partitioned*')
    #client().reducetree('taxi/cleaned*')
    #client().reducetree('taxi/index*')
    #client().reducetree('taxi/projections*')
    #client().reducetree('taxi/raw/projections*')
    import matplotlib.cm as cm
    st = time.time()
    ds = ARDataset()
    ds.partitions()
    #filters = ds.query({'trip_time_in_secs' : [lambda x : (x >= 1999) & (x <= 2000)]})
    filters = None
    global_bounds = ds.gbounds
    local_bounds = global_bounds
    #local_bounds = global_bounds
    local_indexes, (grid_shape, results) = ds.project(
        local_bounds, 'pickup_latitude', 'pickup_longitude', filters
    )
    lxdim1, lxdim2, lydim1, lydim2 = local_indexes
    ed = time.time()
    import pylab
    grid = KSXChunkedGrid(results, grid_shape[-1])
    output = grid.get(0, grid_shape[0] - 1, 0, grid_shape[-1])
    pylab.imshow(output.T[::-1,:] ** 0.3,
                 cmap=cm.Greys_r,
                 extent=global_bounds,
                 interpolation='nearest')
    pylab.figure()
    # st = time.time()
    # output2 = grid.get(lxdim1, lxdim2, lydim1, lydim2)
    # ed = time.time()
    # print ed - st
    # pylab.imshow(output2.T[::-1,:] ** 0.3,
    #              cmap=cm.Greys_r,
    #              extent=local_bounds,
    #              interpolation='nearest')
    # pylab.show()
