import h5py, numpy as np
from os.path import join, exists, dirname, abspath, realpath

import os
count = 0
pwd = dirname(realpath("__file__"))
for i in range(1,86):
    with h5py.File(os.path.join(pwd, 'Embeddings/batch' + str(i) + ".h5") , 'r') as dataall:
        f  = dataall['embed']
        count += len(f)

print("Count : ", str(count))