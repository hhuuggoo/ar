import os
from os.path import relpath, join, basename
from kitchensink import setup_client, client, do, du, dp
setup_client('http://localhost:6323/')
c = client()
c.bc('bootstrap', '/taxi/big.hdf5', data_type='file',_rpc_name='data')
c.execute()
c.br()
