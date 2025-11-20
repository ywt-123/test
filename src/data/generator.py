import keras.utils
import numpy as np
from osgeo import gdal
from os.path import exists, basename

from src.data.preprocess import norm_transforms


class DataGenerator(keras.utils.Sequence):
    def __init__(self, data_path: list, batch_size: int, shuffle=True, name="DataGenerator"):
        super().__init__()

        # store params
        self.data = []
        self.labels = []
        self.labels_name = []

        for data, label in data_path:
            # check if all data and label exists
            flag = False
            for _data in data[0]:
                if not exists(_data):
                    flag = True
            if not exists(label):
                flag = True
            if flag:
                continue
            self.data.append(data)
            self.labels.append(label)

        self.batch_size = batch_size
        self.name = name
        self.shuffle = shuffle
        self.index = np.arange(len(self.labels))

        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.index)

    def __len__(self):
        return len(self.labels) // self.batch_size

    def __getitem__(self, index: int):
        return self._generate_input_data(index)

    def _generate_input_data(self, index: int):
        # 输入数据
        modis_data = []
        era5_data = []
        DEM_data = []
        labels = []
        self.labels_name = []
        base_index = index * self.batch_size

        for batch_num in range(self.batch_size):

            for per_time_list in self.data[self.index[base_index + batch_num]]:

                tif_ndarray = []
                for tif_path in per_time_list[:1]:
                    # print(tif_path)
                    tif_data = gdal.Open(tif_path)
                    temp_ndarray = np.array(tif_data.ReadAsArray(), dtype=np.float32)
                    tif_ndarray.append(temp_ndarray)

                tif_ndarray = np.stack(tif_ndarray)
                # remove NaN or Inf
                invalid_index = np.isnan(tif_ndarray)
                tif_ndarray[invalid_index] = -99
                # tif_ndarray = norm_transforms(tif_ndarray)
                if tif_ndarray.shape != (1, 256, 256):
                    print(per_time_list[0])
                else:
                    modis_data.append(tif_ndarray)

                # print("test")
                nc_ndarray = []
                for nc_path in per_time_list[1:-1]:
                    # print(nc_path)
                    nc_data = gdal.Open(nc_path)
                    temp_ndarray = np.array(nc_data.ReadAsArray(), dtype=np.float32)
                    temp_ndarray = norm_transforms(temp_ndarray, data_type="nc")
                    if temp_ndarray.shape != (256, 256):
                        print(nc_path)
                        temp_ndarray = np.zeros((256, 256))
                    nc_ndarray.append(temp_ndarray)

                nc_ndarray = np.stack(nc_ndarray)
                # remove NaN or Inf
                invalid_index = np.isnan(nc_ndarray)
                nc_ndarray[invalid_index] = -99
                era5_data.append(nc_ndarray)

                DEM_data.append(np.asarray([gdal.Open(per_time_list[-1]).ReadAsArray()], dtype=np.float32))

            # label
            tif_data = gdal.Open(self.labels[self.index[base_index + batch_num]])
            # label name
            _name = basename(self.labels[self.index[base_index + batch_num]])
            self.labels_name.append(_name)

            # 防止标签数据大小不一致
            if tif_data.ReadAsArray().shape != (256, 256):
                print(self.labels[self.index[base_index + batch_num]])
                labels.append(np.zeros((256, 256)))
            else:
                labels.append(np.array(tif_data.ReadAsArray(), dtype=np.float32))

        modis_data = np.asarray(modis_data)
        era5_data = np.asarray(era5_data)
        DEM_data = np.asarray(DEM_data)
        
        # generate landmask
        landmask = DEM_data.copy()
        landmask[landmask > 0.5] = 1.0
        labels = np.asarray(labels)
        
        modis_data = [modis_data[:, x, :, :] for x in range(modis_data.shape[1])]
        era5_data = [era5_data[:, x, :, :] for x in range(era5_data.shape[1])]
        DEM_data = [DEM_data[:, x, :, :] for x in range(DEM_data.shape[1])]
        landmask = [landmask[:, x, :, :] for x in range(landmask.shape[1])]

        # expand dimension
        modis_data = [np.expand_dims(x, axis=-1) for x in modis_data]
        era5_data = [np.expand_dims(x, axis=-1) for x in era5_data]
        DEM_data = [np.expand_dims(x, axis=-1) for x in DEM_data]
        landmask = [np.expand_dims(x, axis=-1) for x in landmask]

        modis_data = modis_data + era5_data + DEM_data + landmask

        return modis_data, labels
