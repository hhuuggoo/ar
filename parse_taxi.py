import os
from os.path import relpath, join, basename, exists
import h5py
import pandas as pd
import cStringIO as StringIO
from kitchensink import setup_client, client, do, du, dp
setup_client('http://power:6323/')
c = client(rpc_name='data', queue_name='data')

datadir = "/data"
path = "/data/taxi"
for root, dirs, files in os.walk(path):
    for f in files:
        if not f.endswith('csv'):
            continue
        path = join(root, f)
        url = relpath(path, datadir)
        c.bc('bootstrap', url, data_type='file', _queue_name='data|power')
c.execute()
c.br()

def to_hdf(df, path):
    f = h5py.File(path, 'a')
    try:
        for x in df.columns:
            col = df[x]
            if col.dtype.kind == 'O':
                col = col.values.astype('str')
            elif col.dtype.kind == 'M':
                col = col.values.astype('int64')
            if x not in f.keys():
                f.create_dataset(x,
                                 data=col,
                                 dtype=col.dtype,
                                 maxshape=(None,),
                                 compression='lzf')
            else:
                ds = f[x]
                orig = ds.shape[0]
                ds.resize((orig + len(df),))
                ds[orig:] = col
    except:
        print 'exception', path, x
        raise
    finally:
        f.close()


def parse(path):
    new_path = path.replace('.csv', '.hdf5')
    if exists(new_path):
        return
    iterobj = pd.read_csv(path,
                          chunksize=500000,
                          skipinitialspace=True,
                          dtype={'store_and_fwd_flag' : 'S4'},
                          parse_dates=['pickup_datetime', 'dropoff_datetime'],
    )
    for idx, df in enumerate(iterobj):
        print idx, 'chunk', new_path
        to_hdf(df, new_path)

def ksparse(obj):
    parse(obj.local_path())
#parse("/data/taxi/trip_data_5.csv")
zips = [du(x) for x in c.path_search('taxi/trip_data*') if '12' not in x]
for z in zips:
    c.bc(ksparse, z)
c.execute()
c.br()
