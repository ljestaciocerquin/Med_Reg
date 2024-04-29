import os
import glob
import argparse
import random
import pandas     as pd
import SimpleITK  as sitk
from   tqdm       import tqdm
from   SimpleITK  import ClampImageFilter
from   processing import cts_operations
from   scipy import ndimage

# Read arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--input_scans_path',    type=str,   default='/data/groups/beets-tan/l.estacio/All_images/')
parser.add_argument('--input_segs_path',   type=str,   default='/data/groups/beets-tan/l.estacio/All_images_seg/')
parser.add_argument('--output_scans_path',   type=str,   default='/data/groups/beets-tan/l.estacio/liver/')
parser.add_argument('--output_segs_path',   type=str,   default='/data/groups/beets-tan/l.estacio/liver_seg/')
args = parser.parse_args()

input_scans_path   = args.input_scans_path
input_segs_path    = args.input_segs_path
output_scans_path  = args.output_scans_path
output_segs_path   = args.output_segs_path


def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print("Folder created in: ", path)
    else:
        print("Folder already created: ", path)
        

def read_data_from_folder(folder_name):
    files = glob.glob(os.path.join(folder_name, '**/*.nii.gz'), recursive=True)
    return files
    
def get_data_to_process(filename):
    data = read_data_from_folder(filename)
    return data
    




def crop_sitkimage(image, lung_mask, dim=(256, 256, 256)):
    """Crops image to around cenre of mass of the mask
    
    Returns coordinates of the patch, cropping the image at center location
    with a given patch size. If the center is in the left or upper border shift
    the center and crop fixed size patch.
    Args:
        center (tuple): (x coordinate, y coordinate)
        image_shape (tuple): shape of image to crop patches from
        dim (int, optional): patch size
    Returns:
        cropped sitk image
    """
    center = ndimage.center_of_mass(sitk.GetArrayFromImage(lung_mask).T)
    center = [int(x) for x in center]
    image_shape = image.GetSize()
    
    # patch half size
    phfx = dim[0] // 2
    phfy = dim[1] // 2
    phfz = dim[2] // 2

    x1 = center[0] - phfx
    x2 = center[0] + dim[0] - phfx
    if x1 < 0:
        x1 = 0
        x2 = dim[0]

    
    
    y1 = center[1] - phfy
    y2 = center[1] + dim[1] - phfy
    if y1 < 0:
        y1 = 0
        y2 = dim[1]

    z1 = center[2] - phfz
    z2 = center[2] + dim[2] - phfz
    if z1 < 0:
        z1 = 0
        z2 = dim[2]


    if x2 > image_shape[0]:
        x2 = image_shape[0]
        x1 = image_shape[0] - dim[0]
    
    if y2 > image_shape[1]:
        y2 = image_shape[1]
        y1 = image_shape[1] - dim[1]
        
    if z2 > image_shape[2]:
        z2 = image_shape[2]
        z1 = image_shape[2] - dim[2]
 
    return image[x1:x2, y1:y2, z1:z2]


def process_scan(scan_path, seg_path, name_to_save_scan, name_to_save_seg):
    clamp = ClampImageFilter()
    clamp.SetUpperBound(300)
    clamp.SetLowerBound(-120)
    
    scan = cts_operations.ReadVolume()(scan_path)
    seg  = cts_operations.ReadVolume()(seg_path)
    scan = crop_sitkimage(scan, seg)    
    scan = cts_operations.TransformFromITKFilter(clamp)(scan)
    scan = cts_operations.ZeroOneScaling()(scan)
    sitk.WriteImage(scan, name_to_save_scan)
    
    seg  = crop_sitkimage(seg, seg)    
    seg  = cts_operations.ZeroOneScaling()(seg)
    sitk.WriteImage(seg, name_to_save_seg)
    
    
def start_preprocessing():
    make_folder(output_scans_path)
    data = get_data_to_process(input_scans_path)
    
    with tqdm(total=len(data)) as pbar:
        for path in data:
            name_scan      = path
            name_to_save_scan =  name_scan.replace(input_scans_path, output_scans_path)#output_scans_path + name_scan.split('/')[-1].split('.')[0] + '.nii.gz'
            folder_to_save_seg = output_segs_path + name_scan.split('/')[-1].split('.')[0] + '/'
            make_folder(folder_to_save_seg)
            name_to_save_seg = folder_to_save_seg + 'liver.nii.gz'
            name_seg = name_to_save_seg.replace(output_segs_path, input_segs_path)
            
            process_scan(name_scan, name_seg, name_to_save_scan, name_to_save_seg)
            pbar.update(1)
            
                
if __name__ == '__main__':
    start_preprocessing()
'''# Example usage:
data = get_data_to_process(input_data_path)
#print(data)
print(data['fix_scan'].iloc[50])
processed_ct_scan = process_nii(data['fix_scan'].iloc[50], "output_image.nii.gz")
print(processed_ct_scan)
processed_ct_scan = sitk.GetImageFromArray(processed_ct_scan)
sitk.WriteImage(processed_ct_scan, r'processed50_pc.nii.gz')'''


