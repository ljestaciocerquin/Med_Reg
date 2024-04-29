import csv
import torch
from model.net import AffineNet
from model.net import ConcatenationElasticNet
from metrics import Similarity_Loss
from metrics import Energy_Loss
from metrics import dice_score
from metrics import compute_dice
from metrics import jacobian_determinant
from utils   import read_test_data
from utils   import visualize_slices
from utils   import save_outputs
from utils   import create_directory
from dataloader import ScanDataLoader
from model.spatial_transformer import SpatialTransformer
from torch.utils.data import DataLoader


def model_init():
    
    # Network definition
    affine_net  = AffineNet(1, [None, None, None], 8, 128, [16, 32, 64, 128])
    elastic_net = ConcatenationElasticNet(1, [None, None, None], 3, 8, [8, 16, 32, 64, 128])
    
    # GPU computation
    device      = torch.device('cuda:0')
    affine_net.to(device)
    elastic_net.to(device)
    
    # Loading the model weights
    aff_chkpt = '/projects/disentanglement_methods/reg_model/plastic_reg_v1_LIVER_end_end/train-plastic/checkpoints/affine_net_120.pth'
    ela_chkpt = '/projects/disentanglement_methods/reg_model/plastic_reg_v1_LIVER_end_end/train-plastic/checkpoints/elastic_net_120.pth'
    affine_state_dict  = torch.load(aff_chkpt)
    elastic_state_dict = torch.load(ela_chkpt)
    
    # Modifying grid size because Spatial transformer has a dynamic shape
    affine_state_dict['spatial_transformer.grid'] = affine_state_dict['spatial_transformer.grid'][:, :, :1, :1, :1]
    elastic_state_dict['spatial_transformer.grid'] = elastic_state_dict['spatial_transformer.grid'][:, :, :1, :1, :1]
    affine_net.load_state_dict(affine_state_dict)
    elastic_net.load_state_dict(elastic_state_dict)
    
    # Loss functions
    nn_loss     = Similarity_Loss() 
    energy_loss = Energy_Loss()
    mse_loss    = torch.nn.MSELoss()
    return affine_net, elastic_net, nn_loss, energy_loss, mse_loss, device


def load_data(data_path):
    
    # Dataset paths
    file_names      = read_test_data(data_path)
    # Testing dataset
    test_dataset    = ScanDataLoader(path_dataset = file_names,
                                input_dim    = [None, None, None],
                                transform    = None)
    
    # Testing dataloader
    test_dataloader = DataLoader(dataset    = test_dataset, 
                                 batch_size = 1, 
                                 shuffle    = False
                                )
    return test_dataloader


def start_test(affine_net, elastic_net, nn_loss, energy_loss, mse_loss, device, test_dataloader, path_to_save_cvs_outputs, path_to_save_nii_outputs):
    
    # Evaluation mode
    affine_net.eval()
    elastic_net.eval()
    
    # Loss for testing
    loss_affine  = 0
    loss_elastic = 0
    
    # Establish convention for real and fake labels during training
    alpha_value  = 1.0
    beta_value   = 1.0
    gamma_value  = 0.1
    
    file_path = path_to_save_cvs_outputs + '/metrics.csv'
    with open(file_path, 'w', newline='') as csvfile:
        fieldnames = ['Index', 'Fixed Scan', 'Moving Scan', 'Dice Affine', 'Dice Elastic', 'Jacobian Affine', 'Jacobian Elastic', 'Affine Loss', 'Elastic Loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, (x_1, x_2, x_3, x_4, x_5) in enumerate(test_dataloader):
            
            # Reading the images and sending to the device (GPU by default)
            fixed       = x_1.to(device)
            moving      = x_2.to(device)
            fixed_seg   = x_3.to(device)
            moving_seg  = x_4.to(device)
            scan_names  = x_5
            fix_name = (str(scan_names[0]).split('/')[-1])[:-3]
            mov_name = (str(scan_names[1]).split('/')[-1])[:-3]
            
            with torch.no_grad():
                st     = SpatialTransformer((256, 256, 256), 'nearest').to(device)
                
                # Forward pass through the Affine model
                out_enc, out_gvp, out_aff  = affine_net(fixed, moving)
                t_0    = out_aff[1].permute(0, 4, 1, 2, 3)
                aff_seg = st(moving_seg, t_0)
                
                
                # (1) Compute affine loss
                cc_aff_loss      = nn_loss.pearson_correlation(fixed, out_aff[2])
                penalty_aff_loss = energy_loss.energy_loss(out_enc[1])
                dice_loss_aff    = dice_score(aff_seg, fixed_seg)
                affine_loss      = cc_aff_loss + penalty_aff_loss + dice_loss_aff
                
                loss_affine  += affine_loss.item()
                print(f'Affine Loss Image {i}: {affine_loss.item()}')
               
                # Forward pass through the Elastic model
                fm, flow_field_ela, elastic_out, ls  = elastic_net(fixed, moving, out_aff[0].detach(), out_aff[1].detach())#elastic_net(fixed, moving, theta.detach(), flow_field_aff.detach())#affine_out.detach(), theta.detach()) 
                t_1    = flow_field_ela.permute(0, 4, 1, 2, 3)
                def_seg = st(moving_seg, t_1)#st(aff_seg, t_1)
                
                # (1) Compute elastic loss
                cc_ela_loss      = nn_loss.pearson_correlation(fixed, elastic_out)
                penalty_ela_loss = energy_loss.energy_loss(flow_field_ela)
                dice_loss_ela    = dice_score(def_seg, fixed_seg)
                elastic_loss     = beta_value*penalty_ela_loss + cc_ela_loss + dice_loss_ela
                loss_elastic    += elastic_loss.item()
                print(f'Elastic Loss Image {i}: {elastic_loss.item()}')
                
                # Applying registration to the segmentations
                dice_loss_aff = compute_dice(fixed_seg.cpu().detach().numpy(), moving_seg.cpu().detach().numpy(), aff_seg.cpu().detach().numpy(), [1])
                dice_loss_ela = compute_dice(fixed_seg.cpu().detach().numpy(), moving_seg.cpu().detach().numpy(), def_seg.cpu().detach().numpy(), [1])
                jac_aff = jacobian_determinant(t_0.permute(0, 1, 3, 4, 2).cpu().detach().numpy())
                jac_ela = jacobian_determinant(t_1.permute(0, 1, 3, 4, 2).cpu().detach().numpy())
                
                visualize_slices(fixed[0], moving[0], out_aff[2][0], out_aff[1][0], elastic_out[0], flow_field_ela[0], i, path_to_save_cvs_outputs)#, elastic_out[0], flow_field_ela[0], i)
                writer.writerow({'Index': i, 'Fixed Scan':fix_name, 'Moving Scan': mov_name, 'Dice Affine': dice_loss_aff, 'Dice Elastic': dice_loss_ela, 'Jacobian Affine': jac_aff, 'Jacobian Elastic': jac_ela, 'Affine Loss': affine_loss.item(), 'Elastic Loss': elastic_loss.item()})
                
                name_to_save = path_to_save_nii_outputs + str(i) + '_'
                save_outputs(fixed, moving, out_aff[1], out_aff[2], flow_field_ela, elastic_out, fixed_seg, moving_seg, aff_seg, def_seg, scan_names, name_to_save)
                
    # Mean loss
    loss_affine  /= len(test_dataloader)
    loss_elastic /= len(test_dataloader)
    print("Testing : loss_affine  = {:.6f},".format(loss_affine))
    print("Testing : loss_elastic = {:.6f},".format(loss_elastic))
    
# Paths
path_to_save_nii_outputs = '/data/groups/beets-tan/l.estacio/out_plastic_reg_v1_LIVER_end_end/'
path_to_save_cvs_outputs = './out_plastic_reg_v1_LIVER_end_end/results/'
create_directory(path_to_save_nii_outputs)
create_directory(path_to_save_cvs_outputs)

# Input data path:
input_data_path = './data/liver/data_pairs_processing_folder.xlsx'
affine_net, elastic_net, nn_loss, energy_loss, mse_loss, device = model_init()
test_dataloader =   load_data(input_data_path)
start_test(affine_net, elastic_net, nn_loss, energy_loss, mse_loss, device, test_dataloader, path_to_save_cvs_outputs, path_to_save_nii_outputs)