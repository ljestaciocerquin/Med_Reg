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
parser.add_argument('--input_data_scans_path',    type=str,   default='/data/groups/beets-tan/l.estacio/liver/')
parser.add_argument('--input_data_segs_path',     type=str,   default='/data/groups/beets-tan/l.estacio/liver_seg/')
parser.add_argument('--anatomy_to_consider',        type=str,   default='/liver.nii.gz')
parser.add_argument('--output_file_path',         type=str,   default='./data/liver/')
args = parser.parse_args()

input_data_scans_path  = args.input_data_scans_path
input_data_segs_path   = args.input_data_segs_path
output_file_path       = args.output_file_path
anatomy_to_consider    = args.anatomy_to_consider


def make_folder(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print("Folder created in: ", path)
    else:
        print("Folder already created: ", path)
        

def get_pairs_source_folder(data):
    random.seed(42) 
    fixed_scans  = []
    moving_scans = []
    fixed_segs   = []
    moving_segs  = []
     
    for scan in data:
        fix_scan =  scan
        mov_scan =  random.choice(data)
        fix_seg  = (fix_scan.replace(input_data_scans_path, input_data_segs_path)).replace('.nii.gz', anatomy_to_consider)
        mov_seg  = (mov_scan.replace(input_data_scans_path, input_data_segs_path)).replace('.nii.gz', anatomy_to_consider)
        fixed_scans.append(fix_scan)
        moving_scans.append(mov_scan)
        fixed_segs.append(fix_seg)
        moving_segs.append(mov_seg)
        
    data    = {'fix_scan': fixed_scans, 'mov_scan': moving_scans, 'fix_seg': fixed_segs, 'mov_seg': moving_segs}
    data_df = pd.DataFrame(data)
    return data_df  


def get_pairs_processing_folder(data):
    random.seed(42) 
    fixed_scans  = []
    moving_scans = []
    fixed_segs   = []
    moving_segs  = []
     
    for scan in data:
        fix_scan =  scan
        mov_scan =  random.choice(data)
        fix_seg  = (fix_scan.replace(input_data_scans_path, input_data_segs_path)).replace('.nii.gz', anatomy_to_consider)
        mov_seg  = (mov_scan.replace(input_data_scans_path, input_data_segs_path)).replace('.nii.gz', anatomy_to_consider)
        fixed_scans.append(fix_scan.replace('/data/groups/beets-tan/', '/processing/'))
        moving_scans.append(mov_scan.replace('/data/groups/beets-tan/', '/processing/'))
        fixed_segs.append(fix_seg.replace('/data/groups/beets-tan/', '/processing/'))
        moving_segs.append(mov_seg.replace('/data/groups/beets-tan/', '/processing/'))
        
    data    = {'fix_scan': fixed_scans, 'mov_scan': moving_scans, 'fix_seg': fixed_segs, 'mov_seg': moving_segs}
    data_df = pd.DataFrame(data)
    return data_df  
                

def read_data_from_folder(folder_name):
    # Use glob to find all .nii.gz files recursively
    files = glob.glob(os.path.join(folder_name, '**/*.nii.gz'), recursive=True)
    return files


def get_data_to_process(filename):
    data = read_data_from_folder(filename)
    return data


def start_preprocessing_segs():
    make_folder(output_file_path)
    data = get_data_to_process(input_data_scans_path)
    data_source = get_pairs_source_folder(data)
    data_proces = get_pairs_processing_folder(data)
    data_source.to_excel(output_file_path + 'data_pairs_source_folder.xlsx', index=False)
    data_proces.to_excel(output_file_path + 'data_pairs_processing_folder.xlsx', index=False)
    
    
if __name__ == '__main__':
    #start_preprocessing_scans()
    start_preprocessing_segs()
    
    
'''# Example usage:
data = get_data_to_process(input_data_scans_path)
#print(data)
print(data['fix_scan'].iloc[50])
processed_ct_scan = process_nii(data['fix_scan'].iloc[50], "output_image.nii.gz")
print(processed_ct_scan)
processed_ct_scan = sitk.GetImageFromArray(processed_ct_scan)
sitk.WriteImage(processed_ct_scan, r'processed50_pc.nii.gz')'''
