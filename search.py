import os
import math
from os.path import relpath, join, basename, exists
import h5py
import numpy as np
import time

import pandas as pd
import cStringIO as StringIO
from kitchensink import setup_client, client, do, du, dp

def get_length(source):
    f = h5py.File(source.local_path(), 'r')
    cols = f.keys()
    return f[cols[0]].shape[0]

def chunks(length, target=500000.):
    num = max(math.ceil(length / target), 2)
    splits = np.linspace(0, length, num).astype('int')
    splits[-1] = length
    output = [(splits[x], splits[x+1]) for x in range(len(splits) - 1)]
    return output

def boolfilter(source, start, end, query_dict, prefilter=None):
    """
    if query_dict is present, we will load the dset into memory, and do the filtering
    if query_dict is not present, we will resort to smart slicing

    prefilter is a boolean vector
    """

    if prefilter is None:
        boolvect = np.ones(end - start, dtype=np.bool)
    else:
        boolvect = prefilter.obj()
    f = h5py.File(source.local_path(), 'r')
    print np.sum(boolvect)
    for field, operations in query_dict.items():
        ds = f[field]
        data = ds[start:end]
        for op in operations:
            val = op(data)
            #print field, op.__name__
            result = boolvect & val
            boolvect = result
            #print np.sum(val), np.sum(result), np.sum(boolvect)
    print 'SUM', np.sum(boolvect)
    obj = do(boolvect)
    obj.save(prefix='index')
    return obj

def smartslice(ds, start, end, boolvect=None):
    if boolvect is None:
        return ds[start:end]
    data = np.empty(np.sum(boolvect), ds.dtype)
    chunksize = ds.chunks[0]
    data_offset = 0
    offset = start
    while offset < end:
        limit = min(offset + chunksize, end)
        subbool = boolvect[offset-start:limit-start]
        if np.sum(subbool) == 0:
            offset = limit
        else:
            sub = ds[offset:limit]
            sub = sub[subbool]
            data_end = data_offset + len(sub)
            data[data_offset:data_end] = sub
            data_offset = data_end
        offset = limit
    return data

def kssmartsliceapply(source, start, end, boolobj, ops=None):
    """
    an op is a func, and a list of fields
    if ops is a list, apply each op to the chunk, and return a list of results
    if it is a tuple, apply the op to the the chunk, return the chunk
    """
    single_return = False
    if isinstance(ops, basestring):
        #no op, which returns one field
        single_return = True
        ops = [(lambda x : x, op)]
    if isinstance(ops, tuple):
        single_return = True
        ops = [ops]
    f = h5py.File(source.local_path(), 'r')
    boolvect = boolobj.obj()
    if np.sum(boolvect) == 0:
        results = [None for op in ops]
    results = []
    for op in ops:
        fields = op[1:]
        op = op[0]
        data = []
        for field in fields:
            ds = f[field]
            result = smartslice(ds, start, end, boolvect)
            data.append(result)
        try:
            result = op(*data)
        except Exception:
            raise Exception(str((source, start, end)))
        results.append(result)
    if single_return:
        results = results[0]
    obj = do(results)
    obj.save(prefix='sliced')
    return obj

class Chunked(object):
    def __init__(self, sources):
        self.sources = sources
        self._lengths = None
        self._chunks = None
    @property
    def lengths(self):
        if self._lengths is not None:
            return self._lengths
        c = client()
        for source in self.sources:
            c.bc(get_length, source)
        c.execute()
        self._lengths = c.br()
        return self._lengths

    @property
    def chunks(self):
        if self._chunks is not None:
            return self._chunks
        self._chunks = list(self.get_chunks())
        return self._chunks

    def get_chunks(self, chunksize=4000000):
        for source, length in zip(self.sources, self.lengths):
            c = chunks(length, target=chunksize)
            for start, end in c:
                yield (source, start, end)

    def query(self, query_dict, prefilter=None):
        c = client()
        if prefilter is None:
            prefilter = [None for x in self.chunks]
        for boolobj, (source, start, end) in zip(prefilter, self.chunks):
            c.bc(boolfilter, source, start, end, query_dict, prefilter=boolobj)
        c.execute()
        return c.br()

    def smartsliceapply(self, slices, ops=None):
        c = client()
        for boolobj, (source, start, end) in zip(slices, self.chunks):
            c.bc(kssmartsliceapply, source, start, end, boolobj, ops=ops)
        c.execute()
        return c.br()

if __name__ == "__main__":
    pass
    # objs = [du(x) for x in c.path_search('taxi/*hdf5')]
    # chunked = Chunked(objs)
    # chunked.chunks
    # def clean_lat(x):
    #     return (x != 0) & ~np.isnan(x) & (x >= 39.) & (x <= 46.)
    # def clean_long(x):
    #     return (x != 0) & ~np.isnan(x) & (x <= -70.) & (x >= -80.)
    # query = chunked.query({'pickup_latitude' : [clean_lat],
    #                        'pickup_longitude' : [clean_long],
    #                        'dropoff_latitude' : [clean_lat],
    #                        'dropoff_longitude' : [clean_long],
    #                    })
    # def helper(lat1, lat2, long1, long2):
    #     bounds = (lat1.max(), lat2.max(), lat1.min(), lat2.min(),
    #               long1.max(), long2.max(), long1.min(), long2.min())

    #     return bounds
    # retval = chunked.smartsliceapply(
    #     query,
    #     ops = (helper, 'pickup_latitude',
    #            'dropoff_latitude',
    #            'pickup_longitude',
    #            'dropoff_longitude',
    #            ))
    # retval = [x.obj() for x in retval]
    # def helper(long1, long2):
    #     return (np.percentile(long1, np.arange(0, 100, 10)),
    #             np.percentile(long2, np.arange(0, 100, 10)))
    # retval = chunked.smartsliceapply(
    #     query,
    #     ops = (helper, 'pickup_longitude', 'dropoff_longitude')
    # )
    # retval = [x.obj() for x in retval]
    # data = []
    # for percentile1, percentile2 in retval:
    #     data.append(percentile1)
    #     data.append(percentile2)
    # data = np.vstack(data)
    # data.mean(axis=0)
    # st = time.time()
    # query = chunked.query({'trip_time_in_secs' : [('>=', 1000),
    #                                               ('<=', 2000)]

    #                    })
    # def helper(x, y):
    #     return x - y
    # retval = chunked.smartsliceapply(query,
    #                                  ops=(helper, 'pickup_latitude', 'pickup_longitude'))
    # print retval
    # ed = time.time()
    # print ed-st

    # st = time.time()
    # c = client()
    # def helper(obj, start, end):
    #     f = h5py.File(obj.local_path())
    #     ds = f['pickup_latitude']
    #     data = ds[start:end]
    #     min = np.min(data)
    #     max = np.max(data)
    #     return min, max
    # for source, start, end in chunked.chunks:
    #     c.bc(helper, source, start, end)
    # c.execute()
    # results = c.br()
    # minval = min([x[0] for x in results])
    # maxval = max([x[1] for x in results])
    # print minval, maxval
    # ed = time.time()
    # print ed - st
