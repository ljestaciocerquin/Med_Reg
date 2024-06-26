import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialTransformer(nn.Module):
    """
        This implementation was taken from:
        https://github.com/voxelmorph/voxelmorph/blob/master/voxelmorph/torch/layers.py
    """
    def __init__(self, size, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids   = torch.meshgrid(vectors)
        grid    = torch.stack(grids)
        grid    = torch.unsqueeze(grid, 0)
        grid    = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)


    def forward(self, src, flow):
        # new locations
        #print('grid: ', self.grid.shape, 'flow: ', flow.shape)
        #print('Flow init ST: ', flow[0, 0, 0, 256, 256])
        new_locs = self.grid + flow
        shape    = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs  = new_locs.permute(0, 2, 3, 1)
            new_locs  = new_locs[..., [1, 0]]
        elif len(shape) == 3: # N x 3 x D x H x W
            new_locs = new_locs.permute(0, 2, 3, 4, 1) # N x D x H x W x 3
            new_locs = new_locs[..., [2, 1, 0]] # To put the axis as (x, y, z)
            # Input to grid_sample: (N, C, D, H, W), (N, D, H, W, 3)
        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
