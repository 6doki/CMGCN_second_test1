import json
import numpy as np
import pandas as pd
import csv

# config = {}
# for filename in ["config/PEMS03.json", "config/STFGNN.json"]:
#     with open(filename, "r") as f:
#         _config = json.load(f)
#         for key in _config:
#             if key not in config:
#                 config[key] = _config[key]
# # print(config)
#
# file_name=config.get("filename","")
# mid_dat = np.load(file_name)
# rawdat = mid_dat[mid_dat.files[0]]
# print(rawdat.shape)


# distance
distance_df_filename="raw_data/PEMS08/distance.csv"
num_of_vertices=170
A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
             dtype=np.float32)
type = 'connectivity'
m,n=0,0
with open(distance_df_filename, 'r') as f:
    f.readline()
    reader = csv.reader(f)
    for row in reader:
        if len(row) != 3:
            continue
        print(row)
        i, j, distance = int(row[0]), int(row[1]), float(row[2])
        print(i,j,distance)
        m=max(i,m)
        n=max(j,n)
        if type == 'connectivity':
            A[i, j] = 1
            A[j, i] = 1
        elif type == 'distance':
            A[i, j] = 1 / distance
            A[j, i] = 1 / distance
        else:
            raise ValueError("type_ error, must be "
                             "connectivity or distance!")
print(len(np.nonzero(A)[0]))
print(np.count_nonzero(A)/(A.shape[0]*A.shape[1]))
print(m,n)


# filename = "raw_data/PEMS04/pems04.npz"
# mid_dat = np.load(filename)
# data = mid_dat[mid_dat.files[0]][:, :, 0]
# if len(data.shape) == 2:
#     data = np.expand_dims(data, axis=-1)
# print(data.shape)

# data = np.load("raw_data/PEMS03/dtw.npy")
# print(data.shape)
# print(data[:10, :10])
