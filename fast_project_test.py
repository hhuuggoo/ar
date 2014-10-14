import pandas as pd
from fast_project import project
import numpy as np

path = "/data/taxi/trip_data_1.csv"

df = pd.read_csv(path,
                 nrows=50000,
                 skipinitialspace=True,
                 dtype={'store_and_fwd_flag' : 'S4'},
                 parse_dates=['pickup_datetime', 'dropoff_datetime'],
)
xmin, xmax, ymin, ymax = (-74.05, -73.75, 40.5, 40.99)
lats = df['pickup_latitude'].values.astype('float64')
longs = df['pickup_longitude'].values.astype('float64')
print lats, longs
marker = np.array([[1]]).astype('float64') #unused, but signature still has it

grid = np.zeros((600,600))
project(longs, lats, grid, xmin, xmax, ymin, ymax, marker)
print np.max(grid)
