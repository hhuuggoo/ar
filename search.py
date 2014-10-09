import os
import math
from os.path import relpath, join, basename, exists
import h5py
import numpy as np
import time

import pandas as pd
import cStringIO as StringIO
from kitchensink import setup_client, client, do, du, dp
setup_client('http://power:6323/')
c = client(rpc_name='data', queue_name='data')

def get_length(source):
    f = h5py.File(source.local_path())
    cols = f.keys()
    return f[cols[0]].shape[0]

def chunks(length, target=500000.):
    num = math.ceil(length / target)
    splits = np.linspace(0, length, num).astype('int')
    splits[-1] = length
    output = [(splits[x], splits[x+1]) for x in range(len(splits) - 1)]
    return output

def boolfilter(source, start, end, query_dict, prefilter=None):
    if prefilter is None:
        boolvect = np.ones(end - start, dtype=np.bool)
    else:
        boolvect = prefilter.obj()
    f = h5py.File(source.local_path())
    for field, operations in query_dict.items():
        ds = f[field]
        data = ds[start:end]
        for operation, value in operations:
            if operation == '>=':
                boolvect = boolvect & (data >= value)
            elif operation == '<=':
                boolvect = boolvect & (data <= value)
            elif operation == '==':
                boolvect = boolvect & (data == value)
    obj = do(boolvect)
    obj.save(prefix='index')
    return obj

def smartslice(ds, start, end, boolvect):
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

def kssmartslice(source, start, end, field,  boolobj, op=None):
    f = h5py.File(source.local_path())
    ds = f[field]
    boolvect = boolobj.obj()
    result = smartslice(ds, start, end, boolvect)
    if op:
        return op(result)
    else:
        obj = do(result)
        obj.save('sliced')
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
        for source, start, end in self.chunks:
            c.bc(boolfilter, source, start, end, query_dict)
        c.execute()
        return c.br()

    def smartslice(self, field, slices, op=None):
        c = client()
        for boolvect, (source, start, end) in zip(slices, self.chunks):
            c.bc(kssmartslice, source, start, end, field, boolvect, op=op)
        c.execute()
        return c.br()

if __name__ == "__main__":
    objs = [du(x) for x in c.path_search('taxi/*hdf5')]
    chunked = Chunked(objs)
    chunked.chunks

    st = time.time()
    query = chunked.query({'trip_time_in_secs' : [('>=', 1000),
                                                  ('<=', 2000)]})
    def helper(arr):
        return len(arr)
    retval = chunked.smartslice('trip_time_in_secs', query, op=helper)
    print sum(retval)
    ed = time.time()
    print ed-st

    st = time.time()
    c = client()
    def helper(obj, start, end):
        f = h5py.File(obj.local_path())
        ds = f['pickup_latitude']
        data = ds[start:end]
        min = np.min(data)
        max = np.max(data)
        return min, max
    for source, start, end in chunked.chunks:
        c.bc(helper, source, start, end)
    c.execute()
    results = c.br()
    minval = min([x[0] for x in results])
    maxval = max([x[1] for x in results])
    print minval, maxval
    ed = time.time()
    print ed - st
