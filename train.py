import os
import torch
import wandb
from   utils                        import create_directory
from   utils                        import cuda_seeds
from   utils                        import weights_init
from   utils                        import read_train_data
from   utils                        import save_images_weights_and_biases_net
from   dataloader                   import ScanDataLoader
#from   early_stopping               import EarlyStopping
from   model.net                    import AffineNet
from   model.net                    import ConcatenationElasticNet
from   model.spatial_transformer    import SpatialTransformer
from   torch.utils.data             import DataLoader
from   tqdm    import tqdm
from   metrics import Similarity_Loss
from   metrics import Energy_Loss
from   metrics import dice_score, dice_loss, ortho_loss, det_loss
from torch import distributed as dist


class Train(object):
    def __init__(self, args):
        
        # Handler
        self.task = args.task
        
        # Data
        self.output_dir  = args.output_dir
        self.data_path   = args.data_path
        self.exp_name_wb = args.exp_name_wb
        self.entity_wb   = args.entity_wb
        
        # Network definition
        self.input_dim   = args.input_dim
        self.input_ch    = args.input_ch
        self.output_ch   = args.output_ch
        self.filters_aff = args.filters_aff
        self.ll_affine   = args.ll_affine
        self.filters_ela = args.filters_ela
        self.group_num   = args.group_num
        
        # Train
        self.seed        = args.seed
        self.num_gpus    = args.num_gpus
        self.start_ep    = args.start_ep
        self.n_epochs    = args.n_epochs
        self.batch_size  = args.batch_size
        self.train_split = args.train_split
        self.alpha_value = args.alpha_value
        self.beta_value  = args.beta_value
        self.gamma_value = args.gamma_value
        self.lr          = args.lr
        self.beta1       = args.beta1
        self.beta2       = args.beta2
        self.re_train    = args.re_train
        self.show_img_wb = args.show_img_wb
        self.chk_aff_to_load = args.chk_aff_to_load
        self.chk_ela_to_load = args.chk_ela_to_load
        
        # Variables to save weights and biases images
        if self.show_img_wb: 
            self.fixed_scan       = None
            self.moving_scan      = None
            self.aligned_aff_scan = None
            self.aligned_ela_scan = None
            self.deformation_field= None
        
        
    def save_outputs(self):
        # Directory to save checkpoints
        self.checkpoints_folder = self.output_dir + self.task + '/' + 'checkpoints/'
        self.results_dir        = self.output_dir + self.task + '/' + 'images-val/'
        create_directory(self.checkpoints_folder)
        create_directory(self.results_dir)
        
        
    def load_model_weights(self):
        # Loading the model weights
        aff_chkpt = self.checkpoints_folder + self.chk_aff_to_load
        ela_chkpt = self.checkpoints_folder + self.chk_ela_to_load
        self.affine_net.load_state_dict(torch.load(aff_chkpt))
        self.elastic_net.load_state_dict(torch.load(ela_chkpt))


    def model_init(self):
        # cuda seeds
        cuda_seeds(self.seed)
        
        # Device
        self.device = torch.device("cuda:0" if (torch.cuda.is_available() and self.num_gpus > 0) else "cpu")
        
        # Network Definitions to the device
        self.affine_net  = AffineNet(self.input_ch, self.input_dim, self.group_num, self.ll_affine, self.filters_aff)
        self.elastic_net = ConcatenationElasticNet(self.input_ch, self.input_dim, self.output_ch, self.group_num, self.filters_ela)
        self.elastic_net.to(self.device)
        self.affine_net.to(self.device)

        if self.re_train:
            print('Loading pretrained weights')
            self.load_model_weights()
        else:
            self.affine_net.apply(weights_init)
            self.elastic_net.apply(weights_init)

        # Handle multi-gpu if desired
        if (self.device.type == 'cuda') and (self.num_gpus > 1):
            self.affine_net = torch.nn.DataParallel(self.affine_net, list(range(self.num_gpus)))
            self.elastic_net = torch.nn.DataParallel(self.elastic_net, list(range(self.num_gpus)))

    
    def init_loss_functions(self):
        self.nn_loss     = Similarity_Loss() 
        self.energy_loss = Energy_Loss()
        #self.disc_loss    = torch.nn.BCELoss()  


    def set_optimizer(self):
        self.aff_optimizer = torch.optim.Adam(self.affine_net.parameters(), lr = self.lr, betas=(self.beta1, self.beta2), weight_decay=1e-5) 
        self.ela_optimizer = torch.optim.Adam(self.elastic_net.parameters(), lr = self.lr, betas=(self.beta1, self.beta2), weight_decay=1e-5) 
        
                    
    def load_dataloader(self):
        # Dataset Path 
        inputs_train                  = read_train_data(self.data_path)
        
        # Training and validation dataset
        train_dataset = ScanDataLoader(inputs_train, self.input_dim, None)
        #valid_dataset = ScanDataLoaderLiver(inputs_valid, self.input_dim, None)
        
        # Training and validation dataloader
        self.train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
        #self.valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=self.batch_size, shuffle=False)
    
    def reduce_mean(self, tensor):
        if dist.is_initialized():
            rt = tensor.clone()
            dist.all_reduce(rt, op=dist.ReduceOp.SUM)
            rt /= dist.get_world_size()
            return rt
        else:
            return tensor
    
    def train_one_epoch(self):
        loss_aff_train  = 0
        loss_ela_train  = 0
        ortho_factor = 0.1
        det_factor = 0.1
        self.affine_net.train()
        self.elastic_net.train()
        optimizer = torch.optim.Adam(list(self.affine_net.parameters()) + list(self.elastic_net.parameters()), lr = self.lr, betas=(self.beta1, self.beta2), weight_decay=1e-5)
        
        for i, (x_1, x_2, x_3, x_4, x_5) in enumerate(self.train_dataloader):
            fixed      = x_1.to(self.device)
            moving     = x_2.to(self.device)
            fixed_seg  = x_3.to(self.device)
            moving_seg = x_4.to(self.device)
            scan_names = x_5
                        
            # Affine Network: [fix_enc, mov_enc], [fix, mov], [theta, transformation, affine_registered_image]  
            #self.aff_optimizer.zero_grad() ------------------------------------- 
            out_enc, _, out_aff  = self.affine_net(fixed, moving)  
            
            # Applying registration to the segmentations
            t_0     = out_aff[1].permute(0, 4, 1, 2, 3)
            st      = SpatialTransformer((256, 256, 256), 'nearest').to(self.device)
            aff_seg = st(moving_seg, t_0)
                
            # Affine loss: cc_aff_loss + alpha*penalty_aff_loss + dice_loss_aff
            '''cc_aff_loss      = self.nn_loss.pearson_correlation(fixed, out_aff[2])
            penalty_aff_loss = self.energy_loss.energy_loss(out_aff[1])#(out_enc[1])
            dice_loss_aff    = dice_score(aff_seg, fixed_seg)
            affine_loss      = cc_aff_loss + self.alpha_value*penalty_aff_loss + dice_loss_aff'''
            A = out_aff[0][..., :3, :3]
            ort = ortho_factor * ortho_loss(A)
            det = det_factor * det_loss(A)
            ort, det = self.reduce_mean(ort), self.reduce_mean(det)
            dice_loss_aff    = dice_loss(fixed_seg, aff_seg)
            affine_loss = ort + det + dice_loss_aff
                
            loss_aff_train  += affine_loss.item()
            #affine_loss.backward() -------------------------------------
            #self.aff_optimizer.step() -------------------------------------
            
            # Elastic Network
            #self.ela_optimizer.zero_grad() -------------------------------------
            #_, flow_field_ela, elastic_out, _  = self.elastic_net(fixed, moving, out_aff[0].detach(), out_aff[1].detach())
            _, flow_field_ela, elastic_out, _  = self.elastic_net(fixed, moving, out_aff[0], out_aff[1])
            
            # Applying registration to the segmentations
            t_1     = flow_field_ela.permute(0, 4, 1, 2, 3)
            #def_seg = st(aff_seg.detach(), t_1)#st(moving_seg, t_1)
            def_seg = st(aff_seg, t_1)#st(moving_seg, t_1)
            
            # Elastic loss: cc_ela_loss + beta*penalty_aff_loss + dice_loss_ela
            cc_ela_loss      = self.nn_loss.pearson_correlation(fixed, elastic_out)
            penalty_ela_loss = self.energy_loss.energy_loss(flow_field_ela)
            dice_loss_ela    = dice_score(def_seg, fixed_seg)
            elastic_loss     = cc_ela_loss + self.beta_value*penalty_ela_loss + dice_loss_ela
            loss_ela_train  += elastic_loss.item()
            #elastic_loss.backward()
            #self.ela_optimizer.step()
            
            
            # ------------ Sum the losses ------------
            total_loss = affine_loss + elastic_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            # Backpropagation
            
            
            
            # Display in weights and biases
            
            it_train_counter = len(self.train_dataloader)
            wandb.log({ 'Iteration': self.epoch * it_train_counter + i, 
                        'ortho_loss': ort.item(),#'cc_aff_loss': cc_aff_loss.item(),
                        'det_loss': det.item(),#'penalty_aff_loss': penalty_aff_loss.item(),
                        'dice_affine': dice_loss_aff.item(),
                        'Train: Affine Loss by Iteration': affine_loss.item(),
                        'cc_ela_loss': cc_ela_loss.item(),
                        'penalty_ela_loss': penalty_ela_loss.item(),
                        'dice_elastic': dice_loss_ela.item(),
                        'Train: Elastic loss by Iteration': elastic_loss.item(),
                        'Train: Total loss by Iteration': total_loss.item(),
                    })
            
            if self.show_img_wb:
                self.fixed_scan       = fixed[0]
                self.moving_scan      = moving[0]
                self.aligned_aff_scan = out_aff[2][0]
                self.aligned_ela_scan = elastic_out[0]
                self.deformation_field= flow_field_ela[0]
            
            if i%50 == 0 and self.show_img_wb:
                epoch_name = str(self.epoch) + "_" + str(i)
                save_images_weights_and_biases_net('Validation Images', self.results_dir, self.fixed_scan, self.moving_scan, self.aligned_aff_scan, self.aligned_ela_scan, self.deformation_field, epoch_name, scan_names)
        
        return loss_aff_train, loss_ela_train
    
    
    def train(self):
        
        # weights and biases
        wandb.init(project=self.exp_name_wb, entity=self.entity_wb)
        #early_stopping = EarlyStopping(patience=100, verbose=True, path=os.path.join(self.checkpoints_folder, 'best_model.pth'))
        
        
        for self.epoch in range(self.start_ep, self.n_epochs):
            # Train
            loss_aff_train, loss_ela_train = self.train_one_epoch()
            
            # Test
            #loss_gen_valid, loss_disc_valid = self.valid_one_epoch()
            
            # Save checkpoints
            if self.epoch % 10 == 0:
                name_ela_net = 'elastic_net_' + str(self.epoch) + '.pth'
                name_aff_net = 'affine_net_' + str(self.epoch) + '.pth'
                torch.save(self.elastic_net.state_dict(), os.path.join(self.checkpoints_folder, name_ela_net))
                torch.save(self.affine_net.state_dict(), os.path.join(self.checkpoints_folder, name_aff_net))
                print('Saving model')
                
            
            # Visualization of images
            #if self.show_img_wb:
            #    save_images_weights_and_biases_net('Validation Images', self.results_dir, self.fixed_scan, self.moving_scan, self.aligned_aff_scan, self.epoch)
                
            # Compute the loss per epoch
            Total_Loss = loss_aff_train + loss_ela_train
            Total_Loss /= len(self.train_dataloader)
            loss_ela_train     /= len(self.train_dataloader)
            loss_aff_train    /= len(self.train_dataloader)
            
            wandb.log({'epoch': self.epoch,
                    'Train: Total loss by epoch (affine)': loss_aff_train,
                    'Train: Total loss by epoch (elastic)': loss_ela_train,
                    'Train: Total loss by epoch': Total_Loss,
                    })
            
            print("Train epoch : {}/{}, loss_affine = {:.6f},".format(self.epoch, self.n_epochs, loss_aff_train)) 
            print("Train epoch : {}/{}, loss_elastic = {:.6f},".format(self.epoch, self.n_epochs, loss_ela_train)) 
            print("Train epoch : {}/{}, total_loss = {:.6f},".format(self.epoch, self.n_epochs, Total_Loss)) 
            
            # early_stopping needs the validation loss to check if it has decresed, 
            # and if it has, it will make a checkpoint of the current model
            # early_stopping(loss_gen_valid, self.aff_net)
            
            '''if early_stopping.early_stop:
                print("Early stopping", self.epoch)
                break'''
            

    def run_train(self):
        # Folders to save outputs
        self.save_outputs()
        
        # Model init
        self.model_init()
        
        # Loss functions
        self.init_loss_functions()
        
        # Optimizers
        self.set_optimizer()
        
        
        # Dataloader
        self.load_dataloader()
        
        
        # Train
        self.train()
        
    