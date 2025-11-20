import os
from datetime import datetime, timedelta
from pprint import pprint

import numpy as np


def get_date_range(start_time : str, end_time : str, day : int = 1):
    """
    获取日期范围内的日期列表。

    参数：
    start_time (str): 开始日期，格式为"YYYY-MM-DD"。
    end_time (str): 结束日期，格式为"YYYY-MM-DD"。

    返回：
    date_list (list): 包含日期范围内所有日期的列表。

    示例：
    >>> get_date_range("2022-01-01", "2022-01-31")
    ['2022-01-01', '2022-01-09', '2022-01-17', '2022-01-25']
    """
    date_list = []
    start_time = datetime.strptime(start_time, "%Y-%m-%d")
    end_time = datetime.strptime(end_time, "%Y-%m-%d")
    
    while (end_time - start_time).days >= 0:
        date_list.append(start_time.strftime("%Y-%m-%d"))
        start_time = start_time + timedelta(days = day)
        
    return date_list


def get_time_step_range(date_range, time_step):
    """
    生成器函数，用于生成给定时间范围内的连续时间步长。

    参数：
    date_range：时间范围，一个列表或可迭代对象。
    time_step：时间步长，一个整数。

    返回：
    生成器对象，每次生成一个时间步长范围的子列表。

    示例：
    >>> date_range = [1, 2, 3, 4, 5]
    >>> time_step = 3
    >>> for step_range in get_time_step_range(date_range, time_step):
    ...     print(step_range)
    [1, 2, 3]
    [2, 3, 4]
    [3, 4, 5]
    """
    for num in range(len(date_range) - time_step + 1):
        yield date_range[num: num + time_step]


num_list = [(350, 6, 5, 0), (350, 7, 6, 0), (350, 19, 4, 1), (350, 20, 5, 1), (350, 21, 6, 1), (350, 22, 7, 1), (350, 23, 8, 1), (350, 32, 3, 2), (350, 33, 4, 2), (350, 34, 5, 2),
            (350, 35, 6, 2), (350, 36, 7, 2), (350, 37, 8, 2), (350, 38, 9, 2), (350, 45, 2, 3), (350, 46, 3, 3), (350, 47, 4, 3), (350, 48, 5, 3), (350, 49, 6, 3),
            (350, 50, 7, 3), (350, 51, 8, 3), (350, 52, 9, 3), (350, 59, 2, 4), (350, 60, 3, 4), (350, 61, 4, 4), (350, 62, 5, 4), (350, 63, 6, 4), (350, 64, 7, 4),
            (350, 65, 8, 4), (350, 66, 9, 4), (350, 67, 10, 4), (350, 72, 1, 5), (350, 73, 2, 5), (350, 74, 3, 5), (350, 75, 4, 5), (350, 76, 5, 5), (350, 77, 6, 5),
            (350, 78, 7, 5), (350, 79, 8, 5), (350, 80, 9, 5), (350, 81, 10, 5), (350, 87, 2, 6), (350, 88, 3, 6), (350, 89, 4, 6), (350, 90, 5, 6), (350, 91, 6, 6),
            (350, 92, 7, 6), (350, 93, 8, 6), (350, 94, 9, 6), (350, 95, 10, 6), (350, 102, 3, 7), (350, 103, 4, 7), (350, 104, 5, 7), (350, 105, 6, 7), (350, 106, 7, 7),
            (350, 107, 8, 7), (350, 108, 9, 7), (350, 109, 10, 7), (350, 116, 3, 8), (350, 117, 4, 8), (350, 118, 5, 8), (350, 119, 6, 8), (350, 120, 7, 8), (350, 121, 8, 8),
            (350, 122, 9, 8), (350, 123, 10, 8), (350, 131, 4, 9), (350, 132, 5, 9), (350, 133, 6, 9), (350, 134, 7, 9), (350, 135, 8, 9), (350, 136, 9, 9), (350, 137, 10, 9),
            (350, 138, 11, 9), (350, 145, 4, 10), (350, 146, 5, 10), (350, 147, 6, 10), (350, 148, 7, 10), (350, 149, 8, 10), (350, 150, 9, 10), (350, 151, 10, 10),
            (350, 152, 11, 10), (350, 153, 12, 10), (350, 159, 4, 11), (350, 160, 5, 11), (350, 161, 6, 11), (350, 162, 7, 11), (350, 163, 8, 11), (350, 164, 9, 11),
            (350, 165, 10, 11), (350, 166, 11, 11), (350, 167, 12, 11), (350, 173, 4, 12), (350, 174, 5, 12), (350, 175, 6, 12), (350, 176, 7, 12), (350, 177, 8, 12),
            (350, 178, 9, 12), (350, 179, 10, 12), (350, 180, 11, 12), (350, 181, 12, 12), (350, 187, 4, 13), (350, 188, 5, 13), (350, 189, 6, 13), (350, 190, 7, 13),
            (350, 191, 8, 13), (350, 192, 9, 13), (350, 193, 10, 13), (350, 194, 11, 13), (350, 195, 12, 13), (350, 196, 13, 13), (350, 200, 3, 14), (350, 201, 4, 14),
            (350, 202, 5, 14), (350, 203, 6, 14), (350, 204, 7, 14), (350, 205, 8, 14), (350, 206, 9, 14), (350, 207, 10, 14), (350, 208, 11, 14), (350, 209, 12, 14),
            (350, 210, 13, 14), (350, 214, 3, 15), (350, 215, 4, 15), (350, 216, 5, 15), (350, 217, 6, 15), (350, 218, 7, 15), (350, 219, 8, 15), (350, 220, 9, 15),
            (350, 221, 10, 15), (350, 222, 11, 15), (350, 223, 12, 15), (350, 226, 1, 16), (350, 227, 2, 16), (350, 228, 3, 16), (350, 229, 4, 16), (350, 230, 5, 16),
            (350, 231, 6, 16), (350, 232, 7, 16), (350, 233, 8, 16), (350, 234, 9, 16), (350, 235, 10, 16), (350, 236, 11, 16), (350, 237, 12, 16), (350, 240, 1, 17),
            (350, 241, 2, 17), (350, 242, 3, 17), (350, 243, 4, 17), (350, 244, 5, 17), (350, 245, 6, 17), (350, 246, 7, 17), (350, 247, 8, 17), (350, 248, 9, 17),
            (350, 249, 10, 17), (350, 250, 11, 17), (350, 253, 0, 18), (350, 254, 1, 18), (350, 255, 2, 18), (350, 256, 3, 18), (350, 257, 4, 18), (350, 258, 5, 18),
            (350, 259, 6, 18), (350, 260, 7, 18), (350, 261, 8, 18), (350, 262, 9, 18), (350, 263, 10, 18), (350, 264, 11, 18), (350, 265, 12, 18), (350, 267, 0, 19),
            (350, 268, 1, 19), (350, 269, 2, 19), (350, 270, 3, 19), (350, 271, 4, 19), (350, 272, 5, 19), (350, 273, 6, 19), (350, 275, 8, 19), (350, 276, 9, 19),
            (350, 277, 10, 19), (350, 281, 0, 20), (350, 282, 1, 20), (350, 283, 2, 20), (350, 284, 3, 20), (350, 285, 4, 20), (350, 286, 5, 20), (350, 296, 1, 21),
            (350, 297, 2, 21), (350, 298, 3, 21), (350, 299, 4, 21), (350, 310, 1, 22), (350, 311, 2, 22), (350, 312, 3, 22), (350, 313, 4, 22), (350, 325, 2, 23),
            (350, 326, 3, 23)]


def get_era5_sequence(base_era5_path: str, year: int, date: str, total: int, num: int, i: int, j: int):
    era5_data = []
    data_type = ["surface_pressure", "temperature_2m", "total_precipitation_sum", "u_component_of_wind_10m", "v_component_of_wind_10m"]

    for _data_type in data_type:
        era5_data.append(f"{base_era5_path}/{year}-{_data_type}/{date}_{_data_type}_{total}_{num}_{i}_{j}.tif")

    return era5_data


def get_MCD_sequence(base_MCD_path: str, year: int, date: str, total: int, num: int, i: int, j: int, data_type: str):
    MCD_data = [f"{base_MCD_path}/{year}-Optical_Depth_{data_type}/{date}_Optical_Depth_{data_type}_{total}_{num}_{i}_{j}.tif"]
    
    return MCD_data



def get_data_sequence(years: list, time_step: int, MCD_type="047", all=False):
    """
    获取数据序列的函数。

    Args:
        years (list): 包含年份的列表。
        time_step (int): 时间步长。
        all (bool, optional): 是否返回全部时间序列。默认为False。

    Returns:
        tuple or list: 如果all为True，则返回全部时间序列的列表。否则返回训练集、验证集和测试集的元组。
    """

    # base_path = "/root/autodl-tmp/data/PM2.5_v2"
    base_path = "D:\BaiduNetdiskDownload\PM2.5_v2"

    # 全部时间序列  [([7][7][7]..., x), ([7][7][7]..., x)....]
    all_time_sequence_lsit = []

    for year in years:

        date_range = get_date_range(f'{year}-01-01', f'{year}-01-01')
        # base_band_path = os.path.join("MOD09A1_cut", str(year))
        base_era5_path = os.path.join(base_path, "ERA5_cut", str(year))
        base_MCD_path = os.path.join(base_path, "MCD19A2_cut", str(year))
        # base_label_path = os.path.join("CHAP_merge_cut_tif", str(year))
        base_label_path = os.path.join("CHAP_cut_tif", str(year))

        for time_step_range in get_time_step_range(date_range, time_step):
            # print(time_step_range)
            for (total, num, i, j) in num_list:

                time_sequence_list = []

                for date in time_step_range:

                    # band_list = []
                    band_list = get_MCD_sequence(base_MCD_path, year, date, total, num, i, j, data_type=MCD_type)

                    band_list = band_list + get_era5_sequence(base_era5_path, year, date, total, num, i, j)
                    # add DEM data
                    band_list = band_list + [f"{base_path}/DEM_cut/china_dem_{total}_{num}_{i}_{j}.tif"]
                    # print(band_list)
                    # exit(0)
                    time_sequence_list.append(band_list)

                date = datetime.strptime(date, "%Y-%m-%d").strftime("%Y%m%d")
                label_name = r'CHAP_PM2.5_D1K_{0}_V4_{1}_{2}_{3}_{4}.tif'.format(date, total, num, i, j)
                temp_path = os.path.join(base_path, base_label_path, label_name)
                # pprint(time_sequence_list)
                all_time_sequence_lsit.append((time_sequence_list, temp_path))
                # break

    # 分割训练集、验证集、测试集 2:1:1
    data_len = len(all_time_sequence_lsit)
    # exit(0)
    if all:
        return all_time_sequence_lsit

    train_sequence = all_time_sequence_lsit[:(data_len // 2)]
    val_sequence = all_time_sequence_lsit[(data_len // 2):(data_len // 2 + data_len // 4)]
    test_sequence = all_time_sequence_lsit[(data_len // 2 + data_len // 4):]

    return train_sequence, val_sequence, test_sequence

# 归一化
def norm_transforms(data: np.ndarray, data_type="tif", tif_values=(-100, 16000), nc_values=(0, 160)) -> np.ndarray:
    """
    对输入数据归一化

    Parameters:
        data (np.ndarray): The input data to be transformed.
        data_type (str): The type of data. Default is "tif".
        tif_values (tuple): The range of values for tif data. Default is (-100, 16000).
        nc_values (tuple): The range of values for nc data. Default is (0, 160).

    Returns:
        np.ndarray: The transformed data.
    """
    if data_type == "nc":
        return (data - nc_values[0]) / (nc_values[1] - nc_values[0])
    else:
        data = (data - tif_values[0]) / (tif_values[1] - tif_values[0])
        return np.clip(data, 0, 1)
