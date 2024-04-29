import torch
import numpy       as     np
from   processing  import cts_operations
from   processing  import cts_processors
from   torch.utils import data 


class ScanDataLoader(data.Dataset):
    def __init__(self,
                 path_dataset: str,
                 input_dim   : int    = [None, None, None],
                 transform   : object = None
                 ):
        self.dataset     = path_dataset
        self.input_shape = tuple(input_dim + [1]) 
        self.transform   = transform
        self.inp_dtype   = torch.float32
        self.loader      = self.__init_operations()

        
    def __init_operations(self):
        return cts_processors.ScanProcessor(
            cts_operations.ReadVolume(),
            cts_operations.ToNumpyArray()
        )
    

    def __len__(self):
        return len(self.dataset)


    def __getitem__(self, index: int):
        
        image_info = self.dataset.iloc[index]
        fixed_scan_path  = str(image_info.squeeze().fix_scan) 
        moving_scan_path = str(image_info.squeeze().mov_scan)
        fixed_seg_path  = str(image_info.squeeze().fix_seg) 
        moving_seg_path = str(image_info.squeeze().mov_seg)
        
        # Set the pixel spacing and slice thickness from the reference image
        scan_image_fix      = self.loader(fixed_scan_path)
        scan_image_mov      = self.loader(moving_scan_path)
        mask_image_fix      = self.loader(fixed_seg_path)
        mask_image_mov      = self.loader(moving_seg_path)
        
        scan_image_fix = torch.from_numpy(scan_image_fix).type(self.inp_dtype)
        scan_image_mov = torch.from_numpy(scan_image_mov).type(self.inp_dtype)
        mask_image_fix = torch.from_numpy(mask_image_fix).type(self.inp_dtype)
        mask_image_mov = torch.from_numpy(mask_image_mov).type(self.inp_dtype)
        
        scan_image_fix = scan_image_fix[None, :]
        scan_image_mov = scan_image_mov[None, :]
        mask_image_fix = mask_image_fix[None, :]
        mask_image_mov = mask_image_mov[None, :]
        
        return scan_image_fix, scan_image_mov, mask_image_fix, mask_image_mov, [fixed_scan_path, moving_scan_path]
