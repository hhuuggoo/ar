import os
from os.path import relpath, join, basename
from kitchensink import setup_client, client, do, du, dp, Client
import cStringIO
import pandas as pd
import numpy as np

setup_client('http://power:6323/')
c = client(rpc_name='data', queue_name='data')

fields = ['posted_date', 'location_1', 'location_2', 'department',
          'title', 'salary', 'start', 'duration', 'job_type',
          'applications', 'company', 'contact', 'phone',
          'fax', 'translated_location', 'latitude',
          'longitude', 'date_first_seen', 'url',
          'date_last_seen']

tsvs = [du(x) for x in c.path_search('*employment*tsv')]
def parse(tsv):
    data = cStringIO.StringIO(tsv.raw())
    raw = pd.read_csv(data, sep="\t",
                      names=fields,
                      parse_dates=['posted_date', 'date_first_seen', 'date_last_seen'],
                      index_col=False)
    return raw

def parse_and_save(tsv):
    raw = parse(tsv)
    raw = raw[['latitude', 'longitude', 'posted_date',
               'date_first_seen', 'date_last_seen',
               'translated_location', 'job_type', 'duration'
           ]]
    url = join('employment', 'pickled', basename(tsv.data_url).replace('.tsv', '.pkl'))
    do(raw).save(url=url)

c.reducetree('employment/pickled/*')
for tsv in tsvs:
    c.bc(parse_and_save, tsv)
c.execute()
c.br()

pickled = [du(x) for x in c.path_search('employment/pickled*')]
def info(df):
    df = df.obj()
    info = {}
    info['length'] = len(df)
    for x in df.columns:
        if df[x].dtype.kind == 'O':
            length = df['job_type'].str.len().dropna().max()
            info[x] = length
    return info

for p in pickled:
    c.bc(info, p)
c.execute()
infos = c.br()
length = sum([x['length'] for x in infos])
duration = max([x['duration'] for x in infos])
job_type = max([x['job_type'] for x in infos])
translated_location = max([x['translated_location'] for x in infos])

dtype = np.dtype([('latitude', 'float64'),
                  ('longitude', 'float64'),
                  ('posted_date', 'int32'),
                  ('date_first_seen', 'int32'),
                  ('date_last_seen', 'int32'),
                  ('translated_location', 'S79'),
                  ('job_type', 'S79'),
                  ('duration', 'S79'),
              ])

import h5py
f = h5py.File('latlong3.hdf5', mode='w')
ds = f.create_dataset('data',
                      shape=(length,),
                      dtype=dtype,
                      compression='lzf')
offset = 0
for p in pickled:
    print p.data_url
    data = p.obj()
    for colname in data.columns:
        col = data[colname]
        if col.dtype.kind == 'M':
            col = col.values.astype('datetime64[D]').max().astype('int16')
        elif col.dtype.kind == 'O':
            col[col.isnull()] = ''
            col = col.values.astype('S79')
        ds[colname, offset:offset+len(data)]  = col
    offset = offset + len(data)

import tables
f = tables.File('table.hdf5', mode='w', filters=tables.Filters(complib='blosc', complevel=9))
t = f.createTable("/","data", description=ds.dtype)
chunksize = 1000000
offset = 0
while offset < ds.shape[0]:
    t.append(ds[offset:offset+chunksize])
    offset += chunksize
