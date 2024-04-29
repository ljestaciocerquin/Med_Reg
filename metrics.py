import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy    as np
import scipy.misc
from   scipy.ndimage import map_coordinates

class Similarity_Loss(nn.Module):

    def __init__(self):
        super(Similarity_Loss, self).__init__()

    def pearson_correlation(self, fixed, warped):
        """
        This loss represents the correlation coefficient loss 
        fixed and warped have shapes (batch, 1, 192, 192, 160)
        """
        # Flatten
        flatten_fixed  = torch.flatten(fixed, start_dim=1)
        flatten_warped = torch.flatten(warped, start_dim=1)

        # Compute the mean
        mean_fixed     = torch.mean(flatten_fixed)
        mean_warped    = torch.mean(flatten_warped)

        # Compute the variance
        var_fixed      = torch.mean((flatten_fixed - mean_fixed) ** 2)
        var_warped     = torch.mean((flatten_warped - mean_warped) ** 2)

        # Compute the covariance
        cov_fix_war    = torch.mean((flatten_fixed - mean_fixed) * (flatten_warped - mean_warped))
        eps            = 1e-6

        # Compute the correlation coefficient loss
        pearson_r      = cov_fix_war / torch.sqrt((var_fixed + eps) * (var_warped + eps))
        raw_loss       = 1 - pearson_r

        return raw_loss
    
    
    def frechet_distance(self, tensor1, tensor2):
        # Flatten the spatial dimensions of 5D tensors to 2D
        tensor1_flat = tensor1.view(tensor1.size(0), tensor1.size(1), -1)
        tensor2_flat = tensor2.view(tensor2.size(0), tensor2.size(1), -1)

        # Compute mean vectors
        mu1 = torch.mean(tensor1_flat, dim=-1)
        mu2 = torch.mean(tensor2_flat, dim=-1)

        # Compute covariance matrices
        diff1 = tensor1_flat - mu1.unsqueeze(-1)
        diff2 = tensor2_flat - mu2.unsqueeze(-1)
        cov1 = torch.matmul(diff1, diff1.transpose(-1, -2)) / (tensor1_flat.size(-1) - 1)
        cov2 = torch.matmul(diff2, diff2.transpose(-1, -2)) / (tensor2_flat.size(-1) - 1)

        # Compute squared Mahalanobis distance
        diff = mu1 - mu2
        sigma_sum = cov1 + cov2 - 2 * torch.sqrt(torch.matmul(torch.matmul(cov1, torch.inverse(cov1 + cov2)), cov2))
        print('Frechet distance sigma_sum: ', sigma_sum)
        # Compute inverse of the sum of covariance matrices across all channels
        sigma_sum_inv = torch.inverse(sigma_sum.sum(dim=1).sum(dim=1).unsqueeze(-1).unsqueeze(-1))
        print('Frechet distance sigma_sum_inv: ', sigma_sum_inv)
        mahalanobis_sq = torch.sum(diff.unsqueeze(-1) @ sigma_sum_inv @ diff.unsqueeze(-2), dim=(1, 2))
        print('Frechet distance mahalanobis_sq: ', mahalanobis_sq)
        # Compute Frechet Distance
        frechet_dist = torch.sqrt(mahalanobis_sq)
        print('Frechet distance: ', frechet_dist)

        return frechet_dist
    
    def feature_maps_loss(self, feature_maps_list):
        
        mse_loss = torch.nn.MSELoss()
        fm_loss  = 0
        fm_frechet_loss = 0
        for fm in feature_maps_list:
            fm_loss += mse_loss(fm[0], fm[1])
            #print('Shapes: ', fm[0].shape, fm[1].shape)
            #fm_frechet_loss += self.frechet_distance(fm[0], fm[1])
        
        fm_loss         /= len(feature_maps_list)
        #fm_frechet_loss /= len(feature_maps_list)
        return fm_loss#, fm_frechet_loss         
            
    
    
    





class Energy_Loss(nn.Module):

    def __init__(self):
        super(Energy_Loss, self).__init__()

    def elastic_loss_2D(self, flow):
        """
        flow has shape (batch, 2, 192, 192)
        Loss for 2D dataset
        """
        dx = (flow[..., 1:, :] - flow[..., :-1, :]) ** 2
        dy = (flow[..., 1:   ] - flow[..., :-1   ]) ** 2
        d = torch.mean(dx) + torch.mean(dy)
        return d / 2.0



    def energy_loss(self, flow):
        """
        This loss represents the total variation loss or the elastic loss
        flow: N x D x H x W x 3 ([1, 256, 160, 192, 3] -> abdomen; [1, 208, 192, 192, 3] -> lung)
        """
        #print('Test 4: ', flow.shape)
        dy   = flow[ :, 1:,  :,  :, :] - flow[ :, :-1, :  , :  , :]
        dx   = flow[ :,  :, 1:,  :, :] - flow[ :, :  , :-1, :  , :]
        dz   = flow[ :,  :,  :, 1:, :] - flow[ :, :  , :  , :-1, :]
        d    = torch.mean(dx**2) + torch.mean(dy**2) + torch.mean(dz**2)
        return d / 3.0



    def energy_loss_(self, flows):
              
        # Compute the Laplacian using 3D convolution with the Laplacian kernel
        laplacian_kernel = torch.tensor([[[[0, 0, 0],
                                        [0, 1, 0],
                                        [0, 0, 0]],
                                        [[0, 1, 0],
                                        [1, -6, 1],
                                        [0, 1, 0]],
                                        [[0, 0, 0],
                                        [0, 1, 0],
                                        [0, 0, 0]]]], dtype=torch.float32).unsqueeze(1).cuda()
        

        # You can apply convolution for each displacement dimension
        laplacian_x = F.conv3d(flows[:, :, :, :, 0].unsqueeze(1), laplacian_kernel, stride=1, padding=1)
        laplacian_y = F.conv3d(flows[:, :, :, :, 1].unsqueeze(1), laplacian_kernel, stride=1, padding=1)
        laplacian_z = F.conv3d(flows[:, :, :, :, 2].unsqueeze(1), laplacian_kernel, stride=1, padding=1)

        # Calculate the squared magnitude of the Laplacians
        laplacian_magnitude = laplacian_x**2 + laplacian_y**2 + laplacian_z**2

        # Sum over all the voxels
        bending_loss = laplacian_magnitude.sum()

        # Calculate the total number of voxels or pixels
        total_voxels = flows.size(0) * flows.size(1) * flows.size(2)  * flows.size(3)

        # Normalize the bending loss
        normalized_bending_loss = bending_loss / total_voxels
        return normalized_bending_loss
    
    
def dice_score(output, target, smooth=1e-6):
    intersection = torch.sum(output * target)
    union        = torch.sum(output) + torch.sum(target)
    dice         = (2. * intersection + smooth) / (union + smooth)
    return dice

def dice_loss(fixed_mask, warped):
    """
    Dice similirity loss
    """

    epsilon = 1e-6

    flat_mask = torch.flatten(fixed_mask, start_dim=1)
    flat_warp = torch.abs(torch.flatten(warped, start_dim=1))
    intersection = torch.sum(flat_mask * flat_warp)
    denominator = torch.sum(flat_mask) + torch.sum(flat_warp) + epsilon
    dice = (2.0 * intersection + epsilon) / denominator

    return 1 - dice


def ortho_loss(A: torch.Tensor):
    eps = 1e-5
    epsI = eps * torch.eye(3).to(A.device)[None]
    C = A.transpose(-1,-2)@A + epsI
    def elem_sym_polys_of_eigenvalues(M):
        M = M.permute(1,2,0)
        sigma1 = M[0,0] + M[1,1] + M[2,2]
        sigma2 = M[0,0]*M[1,1] + M[0,0]*M[2,2] + M[1,1]*M[2,2] - M[0,1]**2 - M[0,2]**2 - M[1,2]**2
        sigma3 = M[0,0]*M[1,1]*M[2,2] + 2*M[0,1]*M[0,2]*M[1,2] - M[0,0]*M[1,2]**2 - M[1,1]*M[0,2]**2 - M[2,2]*M[0,1]**2
        return sigma1, sigma2, sigma3
    s1, s2, s3 = elem_sym_polys_of_eigenvalues(C)
    # ortho_loss = s1 + s2/s3 - 6
    # original formula
    ortho_loss = s1 + (1 + eps) * (1 + eps) * s2 / s3 - 3 * 2 * (1 + eps)
    return ortho_loss.sum()


def det_loss(A):
    # calculate the determinant of the affine matrix
    det = torch.det(A)
    # l2 loss
    return torch.sum(0.5 * (det - 1) ** 2)









# Evaluation
def jacobian_determinant(disp):
    _, _, H, W, D = disp.shape
    #print('disp: ', disp.shape)
    #(1, 3, 208, 192, 192)
    
    gradx  = np.array([-0.5, 0, 0.5]).reshape(1, 3, 1, 1)
    grady  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 3, 1)
    gradz  = np.array([-0.5, 0, 0.5]).reshape(1, 1, 1, 3)
    disp = np.pad(disp, ((0, 0), (0, 0), (1, 1), (1, 1), (1, 1)), mode='constant', constant_values=0)
    
    gradx_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradx, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradx, mode='constant', cval=0.0)], axis=1)
    
    grady_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], grady, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], grady, mode='constant', cval=0.0)], axis=1)
    
    gradz_disp = np.stack([scipy.ndimage.correlate(disp[:, 0, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 1, :, :, :], gradz, mode='constant', cval=0.0),
                           scipy.ndimage.correlate(disp[:, 2, :, :, :], gradz, mode='constant', cval=0.0)], axis=1)

    grad_disp = np.concatenate([gradx_disp, grady_disp, gradz_disp], 0)
    
    '''jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = jacobian[0, 0, :, :, :] * (jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]) -\
             jacobian[1, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :]) +\
             jacobian[2, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :])
    '''
    # Adjust padding in the convolution operation
    jacobian = grad_disp + np.eye(3, 3).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 1:-1, 1:-1, 1:-1]  # Adjusted subregion extraction
    
    # Calculate the Jacobian determinant for each voxel
    jacdet = jacobian[0, 0, :, :, :] * (jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]) -\
             jacobian[1, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :]) +\
             jacobian[2, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :])
    
    
    
    return np.mean(jacdet)

def compute_tre(fix_lms, mov_lms, disp, spacing_fix, spacing_mov):
    
    fix_lms_disp_x = map_coordinates(disp[:, :, :, 0], fix_lms.transpose())
    fix_lms_disp_y = map_coordinates(disp[:, :, :, 1], fix_lms.transpose())
    fix_lms_disp_z = map_coordinates(disp[:, :, :, 2], fix_lms.transpose())
    fix_lms_disp = np.array((fix_lms_disp_x, fix_lms_disp_y, fix_lms_disp_z)).transpose()

    fix_lms_warped = fix_lms + fix_lms_disp
    
    return np.linalg.norm((fix_lms_warped - mov_lms) * spacing_mov, axis=1)


def compute_dice_coefficient(mask_gt, mask_pred):
  """Computes soerensen-dice coefficient.

  compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
  and the predicted mask `mask_pred`.

  Args:
    mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
    mask_pred: 3-dim Numpy array of type bool. The predicted mask.

  Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN.
  """
  volume_sum = mask_gt.sum() + mask_pred.sum()
  if volume_sum == 0:
    return 0
  volume_intersect = (mask_gt & mask_pred).sum()
  
  return 2*volume_intersect / volume_sum

def compute_dice(fixed,moving,moving_warped,labels):
    dice = []
    for i in labels:
        if ((fixed==i).sum()==0) or ((moving==i).sum()==0):
            dice.append(np.NAN)
        else:
            dice.append(compute_dice_coefficient((fixed==i), (moving_warped==i)))
    mean_dice = np.nanmean(dice)
    return mean_dice, dice