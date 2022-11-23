import dataclasses
from typing import List
import os
import copy
import numpy as np
from torch.utils.data import Dataset
import torch
import random

USR_SEL_IDX = 3
USR_READ_IDX = 4


"""
Dataclass to hold a single sample of positional data. 
Only the reads and usr_pos are needed for model training.
"""
@dataclasses.dataclass
class PosData():
    fp: str
    usr_pos: float 
    usr_read: float
    pos: np.ndarray
    reads: np.ndarray
    min_pos: float
    max_pos: float


class PosDataDataset(Dataset):
    """
    Torch dataset to for arm positional data. Stores data
    via the PosData objects, but outputs the [item index, reading, selected position] 
    in order to enable automatic batching. To refer back to the whole object, extract
    the index.

    Parameters:
        dataset_fp: file path of the raw data files in the format used by 12idb. 
            Data will be automatically imported, rescaled, and interpolated.
        
        num_readings: the number of points to interpolate the scans to on import

        import_scans: set to a list of PosData to import directly into the dataset. Ignores
            dataset_fp and num_readings if not None.
    """
    def __init__(self, dataset_fp: str, num_readings: int, import_scans=None):
        if import_scans is None:
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
        else:
            self.scans = import_scans 
    
    def __len__(self) -> int:
        return len(self.scans)

    def __getitem__(self, idx: int): 
        scan = self.scans[idx] 
        return idx, torch.Tensor(scan.reads), torch.Tensor([scan.usr_pos])


def read_pos_dat(dat_fp: str) -> PosData:
    """
    Function to read in a single data file from the 11idb format. 
    Does NOT do any interpolation or rescaling.
    """
    with open(dat_fp, 'r') as dat_f:
        pos_readings = []
        usr_pos = -1
        usr_read = -1
        for i, line in enumerate(dat_f):
            if i == 0:
                usr_pos = float(line.split(' ')[USR_SEL_IDX][:-1].strip())
                usr_read = float(line.split(' ')[USR_READ_IDX][:-1].strip())
            else:
                pos_readings.append([float(raw_num.strip()) for raw_num in line.split(',')])
        pos_readings = np.asarray(pos_readings)
    return PosData(dat_fp, usr_pos, usr_read, pos_readings[:,0], pos_readings[:,1], np.min(pos_readings[:,0]), np.max(pos_readings[:,0]))


# To read data with any format
def read_dat(dat_fp: str) -> PosData:
    with open(dat_fp, 'r') as dat_f:
        pos_readings = []
        usr_pos = -1
        usr_read = -1
        for i, line in enumerate(dat_f):
            if i==0 and line[0]=="%":
                usr_pos = float(line.split(' ')[USR_SEL_IDX][:-1].strip())
                usr_read = float(line.split(' ')[USR_READ_IDX][:-1].strip())
            else:
                pos_readings.append([float(raw_num.strip()) for raw_num in line.split(',')])
        pos_readings = np.asarray(pos_readings)
    return PosData(dat_fp, usr_pos, usr_read, pos_readings[:,0], pos_readings[:,1], np.min(pos_readings[:,0]), np.max(pos_readings[:,0]))


def set_dat(x:float, y:float) -> PosData:
    scan = PosData("", -1, -1, x, y, np.min(x), np.max(x))
    return scan


def get_all_data_fp(dataset_fp: str) -> List[str]:
    """
    Scans through the file structures of data provided by 11idb and 
    extracts the filepaths of all positional data files.
    """
    all_scans = []
    for user_path in os.listdir(dataset_fp):
        scan_list = list(os.listdir(os.path.join(dataset_fp, user_path, 'Scans')))
        if len(scan_list) > 4:
            for scan in scan_list:
                all_scans.append(os.path.join(dataset_fp, user_path, 'Scans', scan))
    return all_scans


def rescale_scan(scan: PosData, num_readings: int) -> PosData:
    """
    Normalizes interpolates posdata for neural network processing.

    Normalizes the arm range between -1, 1
    Normalizes the luminosity readings between -1, 1
    Interpolates the luminosity to num_readings points
    """
    pos_interp = np.linspace(scan.pos[0], scan.pos[-1], num_readings)
    read_interp = np.interp(pos_interp, scan.pos, scan.reads)
    

    # Norm between -1, 1
    usr_val_interp = 2 * ((scan.usr_pos - np.min(pos_interp))/np.ptp(pos_interp)) - 1  
    usr_read_interp = 2 * ((scan.usr_read - np.min(read_interp))/np.ptp(read_interp)) - 1  
    pos_interp = 2 * ((pos_interp - np.min(pos_interp))/np.ptp(pos_interp)) - 1
    read_interp = 2 * ((read_interp - np.min(read_interp))/np.ptp(read_interp)) - 1

    return PosData(scan.fp, usr_val_interp, usr_read_interp, pos_interp, read_interp, scan.min_pos, scan.max_pos)

# scale scan.pos back to the user's scale.
def scaleback(scan: PosData, newpos: float) -> float:
    return (newpos+1)/2*(scan.max_pos-scan.min_pos)+scan.min_pos


def roll_augment_data(ds_subset, dataset, num_augs, pot_rot):
    """
    Performs data augmentation by shifting the luminosity readings left or right
    by a set amount and changing the user position accordingly.

    Params:
        ds_subset: torch data split to perform augmentation on

        dataset: base torch dataset to pull the raw data from

        num_augs: number of augmented samples to make for each sample in the base
            dataset. Output size will be len(ds_subset) * num_augs

        pot_rot: list of ints describing the potential positions the sample can be 
            rotated into. Positions of augmented samples are chosen uniformly with no overlap
    """
    base_scans = [dataset.scans[i] for i in ds_subset.indices]
    aug_set = []
    for scan in base_scans:
        aug_set.append(scan)
        rotations = random.sample(pot_rot, num_augs)
        for rot in rotations:
            reads = np.roll(scan.reads, rot)
            usr_pos = scan.usr_pos
            usr_pos += (rot / len(scan.pos)) * 2
            
            if usr_pos > 1.0:
                usr_pos -= 1
            elif usr_pos < -1.0:
                usr_pos += 1
            aug_set.append(PosData(scan.fp, usr_pos, scan.usr_read, scan.pos, reads, np.min(scan.pos), np.max(scan.pos)))

    return PosDataDataset('', 200, aug_set)