from astropy.table import Table
import numpy as np
import numba
import matplotlib.pyplot as plt

# the different columns in our dataset:
# ra_gal: galaxy right ascension (degrees)
# dec_gal: galaxy declination (degrees)
# ra_gal_mag: magnified galaxy right ascension (degrees)
# dec_gal_mag: magnified galaxy declination (degrees)
# kappa: convergence
# gamma1: shear
# gamma2: shear
# z_cgal: galaxy true redshift
# z_cgal_v: galaxy observed redshift
# unique_gal_id: unique galaxy id
# lmstellar: logarithm of the stellar mass
columns = ['ra_gal', 'dec_gal', 'ra_gal_mag', 'dec_gal_mag', 'kappa', 'gamma1',
        'gamma2', 'z_cgal', 'z_cgal_v', 'unique_gal_id', 'lmstellar']


# size is the size of one axis of the map, in this case 16384
size = 2**14
def get_map(size, index):
    # we create an empty map filled with zeros, to which we add the correct values
    # when going through the dataset
    map = np.zeros((size, size, 10), dtype=np.float64)

    @numba.njit
    def map_to_place(ra_index, dec_index, z_index, data, size):
        temp_map = np.zeros((size, size, 10), dtype=np.float64)
        for i in np.arange(len(data)):
            if i % 1000000 == 0:
                print(i)
            x, y, z = ra_index[i], dec_index[i], z_index[i]
            # dump[x, y, z] = 10**(data[i, 10]) 
            temp_map[x, y, z] = data[i, 4]
        return temp_map.astype(np.float64)

    def data_to_map(data, size):
        between = (0 < data[:, 2]) * (data[:, 2] < 90) * (0 < data[:, 3]) * (data[:, 3] < 90) * (0.07296 < data[:, 7]) * (data[:, 7] < 1.41708)
        data = data[np.where(between)]
        ra_index = (data[:, 2] / 90 * size).astype(np.int32)
        dec_index = (data[:, 3] / 90 * size).astype(np.int32)
        d_z = 1.41708 - 0.07296
        z_index = ((data[:, 7]-.07296) / d_z * 10).astype(np.int32)
        return map_to_place(ra_index, dec_index, z_index, data, size)

    dat = Table.read("/home/ignace/datasets/8336.fits", format='fits', memmap=True)
    batch_size = 50_000_000
    for i in range(500_000_000 // batch_size):
        print(i)
        file = dat[i * batch_size:(i + 1) * batch_size].to_pandas()
        result = data_to_map(file.to_numpy(), size)
        map += result
        
    np.save("/home/ignace/datasets/"+str(size)+"x"+str(size)+"x10_real_"+columns[index]+".npy", map)

    plt.imshow(map.mean(2))
    plt.show()


get_map(size, 4)
