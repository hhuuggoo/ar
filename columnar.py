import h5py
import numpy as np

f = h5py.File('latlong3.hdf5', mode='r')
ds = f['data']
f2 = h5py.File('columnar2.hdf5', mode='w')
lats = ds['latitude']
longs = ds['longitude']
selector = ~(np.isnan(lats) | np.isnan(longs) | lats == 0 | longs == 0)

for idx, col in enumerate(ds.dtype.names):
    print col
    f2.create_dataset(col,
                      data=ds[col][selector],
                      compression='lzf')
