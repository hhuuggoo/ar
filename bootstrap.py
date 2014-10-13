import os
from os.path import relpath, join, basename
from kitchensink import setup_client, client, do, du, dp
setup_client('http://power:6323/')
c = client(rpc_name='data', queue_name='data')
datadir = "/data"
path = "/data/taxi"
for root, dirs, files in os.walk(path):
    for f in files:
        path = join(root, f)
        url = relpath(path, datadir)
        c.bc('bootstrap', url, data_type='file', _queue_name='data|power2', _rpc_name='data')
c.execute()
c.br()
