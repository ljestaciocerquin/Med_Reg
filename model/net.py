import torch
import torch.nn as nn
from collections import OrderedDict
from model.spatial_transformer import SpatialTransformer
import torch.nn.functional as F


def conv_layer(dim_conv: int):
    if dim_conv   == 2:
        return nn.Conv2d
    elif dim_conv == 3:
        return nn.Conv3d


def conv_gl_avg_pool_layer(dim_conv: int):
    if dim_conv   == 2:
        return nn.AdaptiveAvgPool2d
    elif dim_conv == 3:
        return nn.AdaptiveAvgPool3d
    
    
class Encoder(nn.Module):
    def __init__(self,
                input_ch  : int = 1,
                input_dim : int = [None, None, None],
                group_num : int = 8,
                filters   : object = [32, 64, 128, 256] 
                ):
        super(Encoder, self).__init__()
        """
        Inputs:
            - input_dim  : Dimensionality of the input 
            - latent_dim : Dimensionality of the latent space (Z)
            - groups     : Number of groups in the normalization layers
            - filters    : Number of channels or filters to use in the convolutional convolutional layers
        """
        self.input_ch   = input_ch
        self.input_dim  = input_dim
        self.group_num  = group_num
        self.filters    = filters
        modules         = OrderedDict()
        
        for layer_i, layer_filters in enumerate(filters, start=1):
            modules['encoder_block_' + str(layer_i)] = nn.Sequential(
                conv_layer(len(input_dim))(
                    in_channels=input_ch, out_channels=layer_filters, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(num_groups=group_num, num_channels=layer_filters),
                nn.ReLU(inplace=True)
            )
            input_ch = layer_filters
        self.conv_net = nn.Sequential(modules)


    def forward(self, x):
        x = self.conv_net(x)
        return x
    
    

class AffineNet(nn.Module):
    
    def __init__(self,
                 in_channels: int    = 1,
                 in_dim     : int    = [192, 192, 160],
                 group_num  : int    = 8,
                 ll_dim     : int    = 128,
                 filters    : object = [16, 32, 64, 128]
                ): 
        super(AffineNet, self).__init__()
        self.in_channels = in_channels
        self.in_dim      = in_dim
        self.filters     = filters
        self.group_num   = group_num
        self.fts_ll_dim  = ll_dim

        # Encoder Block
        self.encoder = Encoder(self.in_channels, self.in_dim, self.group_num, self.filters)
        
        # Applying global average pooling to have the same feature map dimensions
        self.global_avg_pool = nn.Sequential(OrderedDict([
            ('affine_gl_avg_pool', conv_gl_avg_pool_layer(len(self.in_dim))(output_size=1)), 
        ]))
        
        # Last layer to get the affine transformation parameters
        self.last_layer  = nn.Sequential(OrderedDict([
            ('affine_fts_vec_all' , nn.Flatten()),
            ('affine_last_linear', nn.Linear(in_features=self.filters[-1]*2, out_features=self.fts_ll_dim, bias=False)),
            ('affine_last_act_fn', nn.ReLU(inplace=True)), 
        ]))
        
        # Affine Transformation Blocks
        self.dense_w = nn.Sequential(OrderedDict([
            ('affine_w_matrix'       , nn.Linear(in_features=self.fts_ll_dim, out_features=len(self.in_dim)**2, bias=False)), 
        ]))
        
        self.dense_b = nn.Sequential(OrderedDict([
            ('affine_b_vector'       , nn.Linear(in_features=self.fts_ll_dim, out_features=len(self.in_dim), bias=False)), 
        ]))
        
        # Spatial Transformer
        self.spatial_transformer = SpatialTransformer([1, 1, 1])#(self.in_dim)
        
    
    def forward(self, fixed: torch.tensor, moving: torch.tensor):
        
        # Encoding block
        fix_enc = self.encoder(fixed)
        mov_enc = self.encoder(moving)
        fix = self.global_avg_pool(fix_enc)
        mov = self.global_avg_pool(mov_enc)
        x   = torch.cat((fix, mov), dim=1)
        
        # Get transformation 
        x  = self.last_layer(x)
        w  = self.dense_w(x).view(-1, len(self.in_dim), len(self.in_dim))
        b  = self.dense_b(x).view(-1, len(self.in_dim))
        theta = torch.cat((w, b.unsqueeze(dim=1)), dim=1)
        
        if len(self.in_dim)   == 2:
            theta = theta.view(-1, 2, 3)
        elif len(self.in_dim) == 3:
            theta = theta.view(-1, 3, 4)
        else:
            NotImplementedError('only support 2d and 3d')
            
        # Input to F.affine_grid should be theta(N x 3 x 4) and size(N x C x D x H x W)
        # the output is N × D x H × W × 3 (x, y, z?)
        transformation = F.affine_grid(theta, moving.size(), align_corners=False) 
        if len(self.in_dim)   == 2:
            flow = transformation.permute(0, 3, 1, 2)
        elif len(self.in_dim) == 3:
            flow = transformation.permute(0, 4, 1, 2, 3) # N x 3 x D x H x W
        else:
            NotImplementedError('only support 2d and 3d')
        
        # Update the size of the spatial transformer dynamically
        self.spatial_transformer = SpatialTransformer(moving.shape[2:]).to('cuda')
        affine_registered_image  = self.spatial_transformer(moving, flow)
        
        return [fix_enc, mov_enc], [fix, mov], [theta, transformation, affine_registered_image]  
    


    
'''from torchsummary import summary    
model =  AffineNet( in_channels = 1,
                    in_dim      = [192, 192, 40],
                    filters     = [16, 32, 64, 128],
                    group_num   = 8,
                    ll_dim      = 128)
summary = summary(model.to('cuda'), [(1, 192, 192, 20), (1, 192, 192, 40)])
'''

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, group_dim, dim, use_bias=False): 
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            conv_layer(dim)(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.GroupNorm(num_groups=group_dim, num_channels=out_channels),
            nn.ReLU        (inplace=True),
            conv_layer(dim)(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.GroupNorm(num_groups=group_dim,  num_channels=out_channels),
            nn.ReLU        (inplace=True)
        )

    def forward(self, x):
        out = self.double_conv(x)
        return out  


def max_pool_layer(dim_conv: int):
    if dim_conv   == 2:
        return nn.MaxPool2d
    elif dim_conv == 3:
        return nn.MaxPool3d
     
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, group_dim, dim, use_bias=False):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            max_pool_layer(dim)(kernel_size=2, stride=2),
            DoubleConv(in_channels=in_channels, out_channels=out_channels, group_dim=group_dim, dim=dim, use_bias=use_bias)
        )
    
    def forward(self, x):
        out = self.maxpool_conv(x)
        return out

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, dim, use_bias=False):
        super(OutConv, self).__init__()
        self.conv = conv_layer(dim)(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=use_bias)
    
    def forward(self, x):
        out = self.conv(x)
        return out
    
def conv_up_layer(dim_conv: int):
    if dim_conv   == 2:
        return nn.ConvTranspose2d
    elif dim_conv == 3:
        return nn.ConvTranspose3d

def up_sample_mode(dim_conv: int):
    if dim_conv   == 2:
        return 'nearest'
    elif dim_conv == 3:
        return 'trilinear'

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, group_dim, dim, bilinear=False, use_bias=False):
        super(Up, self).__init__()
        if bilinear:
            self.up  = nn.Sequential(
                nn.Upsample(scale_factor=2, mode=up_sample_mode(dim), align_corners=True),
                conv_layer(dim)(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=use_bias),
                nn.GroupNorm(num_groups=group_dim, num_channels=out_channels),
                nn.ReLU(inplace=True)
            )
            #self.up   = nn.Upsample(scale_factor=2, mode=up_sample_mode(dim), align_corners=True)
            self.conv= DoubleConv(in_channels=in_channels, out_channels=out_channels, group_dim=group_dim, dim=dim, use_bias=use_bias)
        else:
            self.up   = conv_up_layer(dim)(in_channels=in_channels, out_channels=in_channels//2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels=in_channels, out_channels=out_channels, group_dim=group_dim, dim=dim, use_bias=use_bias)
        
    def forward(self, x1, x2):
        x1     = self.up(x1)
        diff_z = x2.size()[2] - x1.size()[2]
        diff_y = x2.size()[3] - x1.size()[3]
        diff_x = x2.size()[4] - x1.size()[4]
        
        x1  = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                         diff_y // 2, diff_y - diff_y // 2,
                         diff_z // 2, diff_z - diff_z // 2])
        x   = torch.cat([x2, x1], dim=1)
        out = self.conv(x)
        return out


class ConcatenationElasticNet(nn.Module):
    def __init__(self,
                 input_ch  : int = 1,
                 input_dim : int = [192, 192, 300],
                 output_ch : int = 3,
                 group_dim : int = 8,
                 filters   : object = [4, 8, 16, 32, 64]
                ):
        super(ConcatenationElasticNet, self).__init__()
        self.input_ch    = input_ch
        self.input_dim   = input_dim
        self.output_ch   = output_ch
        self.filters     = filters
        
        self.convnet     = nn.ModuleDict()
        seq              = self.filters
        
        # ---------------------------------------------------------------- Elastic Network  ---------------------------------------------------------------- 
        # Encoder Block
        # Convolutions layers out_channels, group_dim
        self.convnet['encoder_in'] = DoubleConv(in_channels=self.input_ch, out_channels=self.filters[0], group_dim=group_dim, dim=len(self.input_dim), use_bias=False)
        for i, (j, k) in enumerate(zip(seq, seq[1:]), start=1):  
            self.convnet['encoder_conv_' + str(i)] = Down(in_channels=j, out_channels=k, group_dim=group_dim, dim=len(self.input_dim), use_bias=False)
        
        # Applying global average pooling to have the same feature map dimensions
        self.global_avg_pool = nn.Sequential(OrderedDict([
            ('affine_gl_avg_pool', conv_gl_avg_pool_layer(len(self.input_dim))(output_size=1)), 
        ]))
        
        # Latent space convolution
        self.convnet['latent_space'] = DoubleConv(in_channels=self.filters[-1] * 2, out_channels=self.filters[-1]*2, group_dim=group_dim, dim=len(self.input_dim), use_bias=False) #DoubleConv(in_channels=self.filters[-1]*2, out_channels=self.filters[-1], group_dim=group_dim, dim=len(self.input_dim), use_bias=False)
        #self.convnet['latent_space'] = DoubleConv(in_channels=self.filters[-1], out_channels=self.filters[-1], group_dim=group_dim, dim=len(self.input_dim), use_bias=False)
        
        # Decoder Block
        # Up convolution layers
        
        seq = list(reversed(self.filters))
        for i, (j, k) in enumerate(zip(seq, seq[1:]), start=1):
            #self.convnet['decoder_up_conv_' + str(i)] = Up(in_channels=j, out_channels=k, group_dim=group_dim, dim=len(self.input_dim), bilinear=False, use_bias=False) #Up(in_channels=j*2, out_channels=k*2, group_dim=group_dim, dim=len(self.input_dim), bilinear=True, use_bias=False)
            self.convnet['decoder_up_conv_' + str(i)] = Up(in_channels=j*2, out_channels=k*2, group_dim=group_dim, dim=len(self.input_dim), bilinear=True, use_bias=False)
            
        # Last layer
        #self.convnet['decoder_transformation'] = OutConv(in_channels=self.filters[0], out_channels=self.output_ch, dim=len(self.input_dim), use_bias=False)#OutConv(in_channels=self.filters[1], out_channels=self.output_ch, dim=len(self.input_dim), use_bias=False)
        self.convnet['decoder_transformation'] = OutConv(in_channels=self.filters[1], out_channels=self.output_ch, dim=len(self.input_dim), use_bias=False)
        # Spatial Transformer
        self.spatial_transformer = SpatialTransformer([1, 1, 1])#(self.in_dim)
        
    
    def forward(self, fixed, moving, affine_transf, TA):
        
        # Encoder AlignNet: fixed 
        f_ = self.convnet['encoder_in'](fixed)#(x) 
        f1 = self.convnet['encoder_conv_1'](f_) 
        f2 = self.convnet['encoder_conv_2'](f1) 
        f3 = self.convnet['encoder_conv_3'](f2) 
        
        # Latent space: fixed
        f4  = self.convnet['encoder_conv_4'](f3)
        
        # Encoder AlignNet: moving 
        m_ = self.convnet['encoder_in'](moving)#(x) 
        m1 = self.convnet['encoder_conv_1'](m_) 
        m2 = self.convnet['encoder_conv_2'](m1) 
        m3 = self.convnet['encoder_conv_3'](m2) 
        
        # Latent space: moving
        m4  = self.convnet['encoder_conv_4'](m3)
        
        # Deformation field for each feature map
        # In the affine network, the output is N × D x H × W × 3, that's why we have to:
        # permute flow = transformation.permute(0, 4, 1, 2, 3) # N x 3 x D x H x W and pass through the st
        t_  = F.affine_grid(affine_transf, m_.size(), align_corners=False)#True)
        t_  = t_.permute(0, 4, 1, 2, 3)
        self.spatial_transformer  = SpatialTransformer(m_.shape[2:]).to('cuda')
        fm_ = self.spatial_transformer(m_, t_) # input: (N,C,D,H,W)  -  grid: (N, D, H, W, 3)
        
        t1 = F.affine_grid(affine_transf, m1.size(), align_corners=False)
        t1 = t1.permute(0, 4, 1, 2, 3)
        self.spatial_transformer  = SpatialTransformer(m1.shape[2:]).to('cuda')
        fm1  = self.spatial_transformer(m1, t1)

        t2 = F.affine_grid(affine_transf, m2.size(), align_corners=False)
        t2 = t2.permute(0, 4, 1, 2, 3)
        self.spatial_transformer  = SpatialTransformer(m2.shape[2:]).to('cuda')
        fm2  = self.spatial_transformer(m2, t2)
        
        t3 = F.affine_grid(affine_transf, m3.size(), align_corners=False)
        t3 = t3.permute(0, 4, 1, 2, 3)
        self.spatial_transformer  = SpatialTransformer(m3.shape[2:]).to('cuda')
        fm3  = self.spatial_transformer(m3, t3)
        
        t4 = F.affine_grid(affine_transf, m4.size(), align_corners=False)
        t4 = t4.permute(0, 4, 1, 2, 3)
        self.spatial_transformer  = SpatialTransformer(m4.shape[2:]).to('cuda')
        fm4  = self.spatial_transformer(m4, t4)
        
        # Concatenation
        ls = torch.cat((fm4, f4), dim=1)
        ls = self.convnet['latent_space'](ls)
        
        # Decoder AlignNet
        d1 = torch.cat((fm3, f3), dim=1)
        d2 = torch.cat((fm2, f2), dim=1)
        d3 = torch.cat((fm1, f1), dim=1)
        d4 = torch.cat((fm_, f_), dim=1)
        #print('d1: ', d1.shape)
        #print('d2: ', d2.shape)
        #print('d3: ', d3.shape)
        #print('d4: ', d4.shape)
        
        f  = self.convnet['decoder_up_conv_1'](ls, d1)
        f  = self.convnet['decoder_up_conv_2'](f,  d2)
        f  = self.convnet['decoder_up_conv_3'](f,  d3)
        f  = self.convnet['decoder_up_conv_4'](f,  d4)
        deformation_field = self.convnet['decoder_transformation'](f)

       
        # Update the size of the spatial transformer dynamically
        # Flow affine net: N × D x H × W × 3
        #wM = spatialtranform(tA, M)
        #wM = spatialtranform(tD, wM)
        TA = TA.permute(0, 4, 1, 2, 3)
        self.spatial_transformer = SpatialTransformer(moving.shape[2:]).to('cuda')
        registered_image  = self.spatial_transformer(moving, TA)
        registered_image  = self.spatial_transformer(registered_image, deformation_field)
        deformation_field = deformation_field.permute(0, 2, 3, 4, 1) # N x D x H x W x 3
        ##deformation_field = deformation_field[..., [2, 1, 0]]
        #print('Deformation field after: ', deformation_field.shape)
        
        return [(f_, fm_), (f1, fm1), (f2, fm2), (f3, fm3), (f4, fm4)], deformation_field, registered_image, ls

    
'''from torchsummary import summary    
model =  ConcatenationElasticNet( input_ch  = 1,
                              input_dim = [None, None, None],
                              output_ch = 3,
                              group_dim = 8,
                              filters   = [16, 32, 64, 128, 256]
                            )
summary = summary(model.to('cuda'), [(1, 128, 128, 40), (1, 128, 128, 40), (3, 4), (1, 40, 128, 128)])   
print(model)'''

"""
Simple convolution class
"""
class S_Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(S_Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d   (in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=8,      num_channels=out_ch),
            nn.ReLU     (inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class DiscriminatorNetwork(nn.Module):

    def __init__(self):
        super(DiscriminatorNetwork, self).__init__()
        self.filters = [8, 16, 32, 64]

        self.conv1 = S_Conv(1,               self.filters[0])
        self.conv2 = S_Conv(self.filters[0], self.filters[1])
        self.conv3 = S_Conv(self.filters[1], self.filters[2])
        self.conv4 = S_Conv(self.filters[2], self.filters[3])

        # Last Layer
        self.h     = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.dense = nn.Linear(in_features=self.filters[3], out_features=1, bias=False)
        self.act   = nn.Sigmoid()


    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        f_x = x = self.conv4(x)

        # Last Layer
        x = self.h(x)
        x = torch.flatten(x, 1)
        x = self.dense(x)
        x = self.act(x)

        return x, f_x