import dataclasses
from typing import List
import os
import numpy as np
from torch.utils.data import Dataset
import torch

USR_SEL_IDX = 3
USR_READ_IDX = 4

@dataclasses.dataclass
class PosData():
    fp: str
    usr_pos: float 
    usr_read: float
    pos: np.ndarray
    reads: np.ndarray


def read_pos_dat(dat_fp: str) -> PosData:
    with open(dat_fp, 'r') as dat_f:
        pos_readings = []
        for i, line in enumerate(dat_f):
            if i == 0:
                usr_pos = float(line.split(' ')[USR_SEL_IDX][:-1].strip())
                usr_read = float(line.split(' ')[USR_READ_IDX][:-1].strip())
            else:
                pos_readings.append([float(raw_num.strip()) for raw_num in line.split(',')])
        pos_readings = np.asarray(pos_readings)
    return PosData(dat_fp, usr_pos, usr_read, pos_readings[:,0], pos_readings[:,1])


def get_all_data_fp(dataset_fp: str) -> List[str]:
    all_scans = []
    for user_path in os.listdir(dataset_fp):
        scan_list = list(os.listdir(os.path.join(dataset_fp, user_path, 'Scans')))
        if len(scan_list) > 4:
            for scan in scan_list:
                all_scans.append(os.path.join(dataset_fp, user_path, 'Scans', scan))
    return all_scans


def rescale_scan(scan: PosData, num_readings: int) -> PosData:
    pos_interp = np.linspace(scan.pos[0], scan.pos[-1], num_readings)
    read_interp = np.interp(pos_interp, scan.pos, scan.reads)
    

    # Norm between -1, 1
    usr_val_interp = 2 * ((scan.usr_pos - np.min(pos_interp))/np.ptp(pos_interp)) - 1  
    usr_read_interp = 2 * ((scan.usr_read - np.min(read_interp))/np.ptp(read_interp)) - 1  
    pos_interp = 2 * ((pos_interp - np.min(pos_interp))/np.ptp(pos_interp)) - 1
    read_interp = 2 * ((read_interp - np.min(read_interp))/np.ptp(read_interp)) - 1

    return PosData(scan.fp, usr_val_interp, usr_read_interp, pos_interp, read_interp)


class PosDataDatset(Dataset):
    def __init__(self, dataset_fp: str, num_readings: int):
        scans = [read_pos_dat(scan) for scan in get_all_data_fp(dataset_fp)]
        self.scans = []
        self.discards = []

        # Filter bad scan data
        for scan in scans:
            if len(scan.pos) < 1:
                self.discards.append(scan)
            elif scan.usr_pos > np.max(scan.pos) or scan.usr_pos < np.min(scan.pos):
                self.discards.append(scan)
            else:
                self.scans.append(rescale_scan(scan, num_readings))
        
        print(f'Discarded {len(self.discards)} invalid scans!')
    
    def __len__(self) -> int:
        return len(self.scans)

    def __getitem__(self, idx: int): 
        scan = self.scans[idx] 
        return idx, torch.Tensor(scan.reads), torch.Tensor([scan.usr_pos])
