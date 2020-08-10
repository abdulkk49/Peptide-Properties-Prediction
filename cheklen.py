import h5py, numpy as np
count = 0
for i in range(1,86):
    with h5py.File('./Embeddings/batch' + str(i), 'r') as dataall:
        f  = dataall['embed']
        count += len(f)

print("Count : ", str(count))