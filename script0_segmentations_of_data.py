import os
import glob
import argparse
import random
import pandas     as pd
import SimpleITK  as sitk
from   tqdm       import tqdm
from   SimpleITK  import ClampImageFilter
from   processing import cts_operations

# Read arguments 
parser = argparse.ArgumentParser()
parser.add_argument('--input_data_path',    type=str,   default='/data/groups/beets-tan/l.estacio/All_images/')
parser.add_argument('--output_data_path',   type=str,   default='/data/groups/beets-tan/l.estacio/All_images_seg_13/')
args = parser.parse_args()

input_data_path  = args.input_data_path
output_data_path = args.output_data_path


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
    
def process_scan(image_path, output_path):
    ts_cmd = f"TotalSegmentator -i {image_path} -o {output_path} --roi_subset spleen kidney_right kidney_left gallbladder liver stomach pancreas vertebrae_L2 vertebrae_L1 vertebrae_T12 vertebrae_T11 rib_left_12 rib_right_12 --statistics"
    os.system(ts_cmd)


def start_preprocessing():
    make_folder(output_data_path)
    data = get_data_to_process(input_data_path)
    
    with tqdm(total=len(data)) as pbar:
        for path in data:
            name_scan      = path
            folder_to_save = output_data_path + name_scan.split('/')[-1].split('.')[0]
            make_folder(folder_to_save)
            process_scan(name_scan, folder_to_save)
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


