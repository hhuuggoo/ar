import h5py

def uncompress(paths):
    new_path = "big.hdf5"
    path = paths[0]
    f = h5py.File(path, 'r')
    keys = f.keys()
    f.close()
    f2 = h5py.File(new_path, 'a')
    for k in keys:
        lengths = []
        for p in paths:
            f = h5py.File(p, 'r')
            dtype = f[k].dtype
            lengths.append(f[k].shape[0])
            f.close()
        f2.create_dataset(k, dtype=dtype, shape=(sum(lengths), ))
        offset = 0
        for p in paths:
            print p, k
            f = h5py.File(p, 'r')
            data = f[k][:]
            f.close()
            f2[k][offset:offset+len(data)] = data
            offset += len(data)
    f2.close()

uncompress(["/home/bokeh/trip_data_1.hdf5", "/home/bokeh/trip_data_2.hdf5",
            "/home/bokeh/trip_data_3.hdf5", "/home/bokeh/trip_data_4.hdf5",
            "/home/bokeh/trip_data_5.hdf5", "/home/bokeh/trip_data_6.hdf5"])
