

# inputs_size = ((256, 256, 1), )
# # # (256, 256, 1)
# # print(inputs_size[0])

# labels = ["MCD", "sp", "t2m", "total_precipitation", "u10", "v10", "DEM"]
# inputs_list = []
# for label in labels:
#         inputs_list.append(f'{inputs_size},{label}')

# print(inputs_list)



import numpy as np
np_data = np.load(r'D:\BaiduNetdiskDownload\data\output\prediction\CHAP_PM2.5_D1K_20210704_V4_350_6_5_0.tif.npz')
print(np_data['predict'])
print("=======================")
print(np_data['label'])