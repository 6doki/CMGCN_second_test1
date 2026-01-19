import numpy as np

# test_PEMS03_dtw=np.load('./raw_data/PEMS03/dtw.npy')
# test_PEMS03_adj_mx=np.load('./raw_data/PEMS03/adj_mx.pkl')

test_PEMS04_dtw=np.load('./raw_data/PEMS04/dtw.npy')
test_PEMS04_distance=np.load('./raw_data/PEMS04/distance.csv',allow_pickle=True)

# test_PEMS08_dtw=np.load('./raw_data/PEMS08/dtw.npy',allow_pickle=True)
# test_PEMS08_distance=np.load('./raw_data/PEMS08/distance.csv')
# test_PEMS04_10=np.load('./raw_data/PEMS04_10/dtw.npy')


print('-------------------------------------------')
print(test_PEMS04_dtw)
print('-------------------------------------------')
print(test_PEMS04_distance)
print('-------------------------------------------')
print(test_PEMS08_dtw)
print('-------------------------------------------')
print(test_PEMS08_distance)
print('-------------------------------------------')