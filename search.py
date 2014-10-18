import os
import math
from os.path import relpath, join, basename, exists
import h5py
import numpy as np
import time

import pandas as pd
import cStringIO as StringIO
from kitchensink import setup_client, client, do, du, dp
from kitchensink.admin import timethis

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
    for field, operations in query_dict.items():
        with timethis('load_%s' % field):
            ds = f[field]
            data = ds[start:end]
        with timethis('filter_%s' % field):
            for op in operations:
                val = op(data)
                result = boolvect & val
                boolvect = result
    with timethis('saving'):
        obj = do(boolvect, fmt='bloscpickle')
        obj.save(prefix='index')
    return obj

import ardata
def smartslice(ds, start, end, boolvect=None):
    name = ds.name[1:]
    mapped = ardata.mread(name)
    if mapped is not None:
        print "GOT MAPPED", name, start, end
        data = mapped[start:end]
        if boolvect is not None:
            print boolvect.shape, mapped.shape, data.shape
            data = data[boolvect]
        return data
    print "NO MAPPED", name
    if boolvect is None:
        return ds[start:end]
    data = np.empty(np.sum(boolvect), ds.dtype)
    chunksize = end - start
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

lengths = {}
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
        if all([source.data_url in lengths for source in self.sources]):
            return [lengths[source.data_url] for source in self.sources]
        for source in self.sources:
            c.bc(get_length, source)
        c.execute()
        self._lengths = c.br()
        for source, length in zip(self.sources, self._lengths):
            lengths[source.data_url] = length
        return self._lengths

    @property
    def chunks(self):
        if self._chunks is not None:
            print len(self._chunks), "numchunks"
            return self._chunks
        self._chunks = list(self.get_chunks())
        print len(self._chunks), "numchunks"
        return self._chunks

    def get_chunks(self, chunksize=2000000):
        for source, length in zip(self.sources, self.lengths):
            c = chunks(length, target=chunksize)
            for start, end in c:
                yield (source, start, end)


if __name__ == "__main__":
    pass
