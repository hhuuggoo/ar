from kitchensink import settings
import numpy as np

from os.path import join, exists
import h5py
import cPickle
cache = {}

def cache_field(field):
    if not settings.is_server:
        print "NOT SERVER"
        return
    path = join(settings.datadir, "taxi", "big.hdf5")
    f = h5py.File(path, 'r')
    try:
        data = f[field][:]
    finally:
        f.close()
    cache[field] = data

def cached(field):
    if field in cache:
        return cache[field]


def memmap(arr, name):
    info = name + ".info"
    data_name = name + ".nmap"
    with open(info, "w+") as f:
        cPickle.dump(arr.dtype, f, -1)
    mapped = np.memmap(data_name, mode="w+", shape=arr.shape, dtype=arr.dtype)
    mapped[:] = arr
    mapped.flush()

def mread(name):
    info = "/mnt/ramdisk/%s.info" % name
    data_name = "/mnt/ramdisk/%s.nmap" % name
    if not (exists(data_name) and exists(info)):
        print 'could not find'
        return None
    with open(info, "rb") as f:
        dtype = cPickle.load(f)
    mapped = np.memmap(data_name, mode='r', dtype=dtype)
    return mapped
