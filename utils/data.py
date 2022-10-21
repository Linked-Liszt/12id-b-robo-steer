import dataclasses
import torch
from typing import List
import os
import numpy as np

USR_SEL_IDX = 3

@dataclasses.dataclass
class PosData():
    usr_pos: float 
    pos: np.ndarray
    reads: np.ndarray


def read_pos_dat(dat_fp: str) -> PosData:
    with open(dat_fp, 'r') as dat_f:
        pos_readings = []
        for i, line in enumerate(dat_f):
            if i == 0:
                usr_pos = float(line.split(' ')[USR_SEL_IDX][:-1].strip())
            else:
                pos_readings.append([float(raw_num.strip()) for raw_num in line.split(',')])
        pos_readings = np.asarray(pos_readings)
        return PosData(usr_pos, pos_readings[:,0], pos_readings[:,1])


def get_all_data_fp(dataset_fp: str) -> List[str]:
    all_scans = []
    for user_path in os.listdir(dataset_fp):
        scan_list = list(os.listdir(os.path.join(dataset_fp, user_path, 'Scans')))
        if len(scan_list) > 4:
            for scan in scan_list:
                all_scans.append(os.path.join(dataset_fp, user_path, 'Scans', scan))
    return all_scans


def rescale_scan(scan: PosData, num_readings: int) -> PosData:
    pass

