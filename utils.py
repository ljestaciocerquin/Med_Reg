import os
import torch
import wandb
import numpy    as np 
import pandas   as pd
import torch.nn as nn
import SimpleITK as sitk
import torchvision.transforms as T
import matplotlib.pyplot      as plt
from PIL import Image, ImageDraw, ImageFont

def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        print("Directory created in: ", path)
    else:
        print("Directory already created: ", path)


def cuda_seeds(seed):
    # GPU operations have a separate seed we also want to set
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Additionally, some operations on a GPU are implemented stochastic for efficiency
    # We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark    = False
    
    
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv3d") != -1:
        m.weight.data = nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

    elif classname.find("Linear") != -1:
        nn.init.kaiming_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
            
            
def read_train_data(path_input):
    data = pd.read_excel(path_input)
    return data[0:68] # 80% 


def read_test_data(path_input):
    data = pd.read_excel(path_input)
    return data[68:] # 20%


def visualize_slices(fixed, moving, w_0, t_0, w_1, t_1, i, folder_name):
    fixed_copy  = torch.squeeze(fixed.cpu(),  0)
    moving_copy = torch.squeeze(moving.cpu(), 0)
    w_0_copy    = torch.squeeze(w_0.cpu(), 0)
    w_1_copy    = torch.squeeze(w_1.cpu(), 0)
    t_0_copy    = t_0.cpu()
    t_1_copy    = t_1.cpu()
    
    t0_x   = t_0_copy[:, :, :, 0]
    t0_y   = t_0_copy[:, :, :, 1]
    t0_z   = t_0_copy[:, :, :, 2]
    t0_mag = torch.sqrt(t0_x**2 + t0_y**2 + t0_z**2)
    
    t1_x   = t_1_copy[:, :, :, 0]
    t1_y   = t_1_copy[:, :, :, 1]
    t1_z   = t_1_copy[:, :, :, 2]
    t1_mag = torch.sqrt(t1_x**2 + t1_y**2 + t1_z**2)

    f, axarr    = plt.subplots(3,2)
    axarr[0,0].imshow(fixed_copy [200, :, :], cmap='gray')
    axarr[0,0].set_title('Fixed')
    axarr[0,1].imshow(moving_copy[200, :, :], cmap='gray')
    axarr[0,1].set_title('Moving')
    axarr[1,0].imshow(w_0_copy[200, :, :], cmap='gray')
    axarr[1,0].set_title('Affine Output')
    axarr[1,1].imshow(t0_mag[200, :, :], cmap='gray' )
    axarr[1,1].set_title('Affine sampling grid')
    axarr[2,0].imshow(w_1_copy[200, :, :], cmap='gray')
    axarr[2,0].set_title('Elastic Output')
    axarr[2,1].imshow(t1_mag[200, :, :], cmap='gray')
    axarr[2,1].set_title('Deformation field elastic')
    
 
    plt.savefig(os.path.join(folder_name, '{}.png').format(i))
    
    

def save_images_weights_and_biases_net(table_name, path_to_save, fix_scan, mov_scan, pred_aff_align, pred_ela_align, def_field, epoch, name_scans):
    
    t0_x   = def_field[:, :, :, 0]
    t0_y   = def_field[:, :, :, 1]
    t0_z   = def_field[:, :, :, 2]
    t0_mag = torch.sqrt(t0_x**2 + t0_y**2 + t0_z**2)
    
    #PIL VERSION
    transform     = T.ToPILImage() 
    fix_scan     = transform(fix_scan[:,200,:,:].squeeze()).convert("L") 
    moving_scan     = transform(mov_scan[:,200,:,:].squeeze()).convert("L")
    aff_scan = transform(pred_aff_align[:,200,:,:].squeeze()).convert("L") # The 0 is in order to visualize the deformation field transform(w0_img[0,:,:,50].squeeze()).convert("L")
    ela_scan = transform(pred_ela_align[:,200,:,:].squeeze()).convert("L")
    deformation_field = transform(t0_mag[200,:,:].squeeze()).convert("L")
    
    table = wandb.Table(columns=['Fix Scan', 'Mov Scan', 'Affine Scan', 'Elastic Scan', 'Deformation Scan'], allow_mixed_types = True)
    
    fix_scan.show()                              
    fix_scan.save(path_to_save + epoch + "_fix_scan.png")    
    moving_scan.show() 
    moving_scan.save(path_to_save + epoch + "_mov_scan.png")    
    aff_scan.show() 
    aff_scan.save(path_to_save + epoch + "_affine_scan.png")  
    ela_scan.show() 
    ela_scan.save(path_to_save + epoch + "_elastic_scan.png")    
    deformation_field.show()
    deformation_field.save(path_to_save + epoch + "_deformation_scan.png")  
    
    '''table.add_data(
        wandb.Image(Image.open(saving_examples_folder + "fixed_image.png")),
        wandb.Image(Image.open(saving_examples_folder + "moving_image.png")),
        wandb.Image(Image.open(saving_examples_folder + "affine_image.png")),
        wandb.Image(Image.open(saving_examples_folder + "deformation_image.png"))
        #wandb.Image(Image.open(saving_examples_folder + "deformation_field_image.png"))
    )
    
    wandb.log({table_name: table})'''
    
    
def save_outputs(fx, mv, t_0, w_0, t_1, w_1, seg_fix, seg_mov, aff_seg, ela_seg, scan_names, path_to_save):

    fix_name = (path_to_save + str(scan_names[0]).split('/')[-1])[:-3]
    mov_name = (path_to_save + str(scan_names[1]).split('/')[-1])[:-3]
    print(' Fix image: ', fix_name)
    print(' Moving image: ', mov_name)  
      
    t_0_numpy = np.squeeze(t_0.detach().cpu().numpy(), axis=0)          # from [1, 3, 192, 192, 160] -> [3, 192, 192, 160]
    np.save(path_to_save  + 't_0.npy', t_0_numpy)
    
    t0_x   = t_0_numpy[:, :, :, 0]
    t0_y   = t_0_numpy[:, :, :, 1]
    t0_z   = t_0_numpy[:, :, :, 2]
    t0_mag = np.sqrt(t0_x**2 + t0_y**2 + t0_z**2)
    sitk.WriteImage(sitk.GetImageFromArray(t0_mag), path_to_save  + 't_0.nii.gz')
    
    w_0_numpy = np.squeeze(w_0.detach().cpu().numpy(), axis=(0 ,1))     # from [1, 1, 192, 192, 160] -> [192, 192, 160]
    sitk.WriteImage(sitk.GetImageFromArray(w_0_numpy), path_to_save  + 'w_0.nii.gz')
    
    t_1_numpy = np.squeeze(t_1.detach().cpu().numpy(), axis=0)          # from [1, 3, 192, 192, 160] -> [3, 192, 192, 160]
    np.save(path_to_save  + 't_1.npy', t_1_numpy)
    
    t1_x   = t_1_numpy[:, :, :, 0]
    t1_y   = t_1_numpy[:, :, :, 1]
    t1_z   = t_1_numpy[:, :, :, 2]
    t1_mag = np.sqrt(t1_x**2 + t1_y**2 + t1_z**2)
    sitk.WriteImage(sitk.GetImageFromArray(t1_mag), path_to_save  + 't_1.nii.gz')
    
    w_1_numpy = np.squeeze(w_1.detach().cpu().numpy(), axis=(0, 1))     # from [1, 1, 192, 192, 160] -> [192, 192, 160]
    sitk.WriteImage(sitk.GetImageFromArray(w_1_numpy), path_to_save  + 'w_1.nii.gz')

    fx_numpy  = np.squeeze(fx.detach().cpu().numpy(),  axis=(0, 1))     # from [1, 1, 192, 192, 160] -> [192, 192, 160]
    sitk.WriteImage(sitk.GetImageFromArray(fx_numpy), path_to_save  + 'fixed.nii.gz')
    
    mv_numpy  = np.squeeze(mv.detach().cpu().numpy(),  axis=(0, 1))     # from [1, 1, 192, 192, 160] -> [192, 192, 160]
    sitk.WriteImage(sitk.GetImageFromArray(mv_numpy), path_to_save  + 'moving.nii.gz')
    
    fx_seg_numpy  = np.squeeze(seg_fix.detach().cpu().numpy(),  axis=(0, 1))     # from [1, 1, 192, 192, 160] -> [192, 192, 160]
    sitk.WriteImage(sitk.GetImageFromArray(fx_seg_numpy), path_to_save  + 'fix_seg.nii.gz')
    
    mv_seg_numpy  = np.squeeze(seg_mov.detach().cpu().numpy(),  axis=(0, 1))     # from [1, 1, 192, 192, 160] -> [192, 192, 160]
    sitk.WriteImage(sitk.GetImageFromArray(mv_seg_numpy), path_to_save  + 'mov_seg.nii.gz')
    
    aff_seg_numpy  = np.squeeze(aff_seg.detach().cpu().numpy(),  axis=(0, 1))     # from [1, 1, 192, 192, 160] -> [192, 192, 160]
    sitk.WriteImage(sitk.GetImageFromArray(aff_seg_numpy), path_to_save  + 'aff_seg.nii.gz')
    
    ela_seg_numpy  = np.squeeze(ela_seg.detach().cpu().numpy(),  axis=(0, 1))     # from [1, 1, 192, 192, 160] -> [192, 192, 160]
    sitk.WriteImage(sitk.GetImageFromArray(ela_seg_numpy), path_to_save  + 'ela_seg.nii.gz')
    