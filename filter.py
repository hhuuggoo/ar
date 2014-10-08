import h5py
import numpy as np
import scipy as sp
import scipy.sparse as sparse
import scipy.signal as signal
import scipy.ndimage
import matplotlib.cm as cm
import time
import numba
from numpy.core.defchararray import find as npfind
from numpy.core.defchararray import lower as nplower

### grab 2 1d vectors (lat/long) from hdf5 file

f = h5py.File('columnar.hdf5', mode='r')
chunksize = 1000
def chunked_find(col, func):
    offset = 0
    results = []
    while offset < len(col):
        sub = col[offset:offset + chunksize]
        results.append(func(sub))
        offset += chunksize
    return results
st = time.time()
result = chunked_find(f['translated_location'], lambda x : npfind(nplower(x), 'honduras')  > 0)
result = np.hstack(result)
ed = time.time()
print ed-st
st = time.time()
result2 = chunked_find(f['job_type'], lambda x : npfind(nplower(x), 'tiempo')  > 0)
result2 = np.hstack(result2)
ed = time.time()
print ed-st
