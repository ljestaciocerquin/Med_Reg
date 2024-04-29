from   handler import Handler
import argparse

def main(args):
    
    handler = Handler(args=args)
    handler.run()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unsupervised Registration of Medical Images for Prognostic Monitoring')
    
    # Handler 
    parser.add_argument('--task',        type=str,  default='train-plastic',    help='task to be performed train, train-adv, test')
    
    # Data
    parser.add_argument('--output_dir',  type=str, default='/projects/disentanglement_methods/reg_model/plastic_reg_only_liver_end_end_ortho_det/',   help='folder to save all outputs from train | train-adv | test')
    parser.add_argument('--data_path' ,  type=str, default='./data/liver/data_pairs_processing_folder.xlsx',                          help='input data to train the model')
    parser.add_argument('--exp_name_wb', type=str, default='PlasNet',            help='experiment name weights and biases')
    parser.add_argument('--entity_wb',   type=str, default='ljestaciocerquin',   help='Entity for weights and biases')
    parser.add_argument('--show_img_wb', type=bool,  default=True,               help='it shows images when the dataset is open source: True | False')
    
    # Networks
    parser.add_argument('--input_ch',     type=int, default=1,                    help='number of input channels for the affine and deformation networks')
    parser.add_argument('--input_dim',    type=int, default=[None, None, None],   help='image dimension')
    parser.add_argument('--filters_aff',  type=int, default=[16, 32, 64, 128],    help='filters to create the Affine network') #[16, 32, 64, 128]
    parser.add_argument('--filters_ela',  type=int, default=[8, 16, 32, 64, 128], help='filters to create the Elastic network')
    parser.add_argument('--group_num',    type=int, default=8,                    help='group normalization size')
    parser.add_argument('--ll_affine',    type=int, default=128,                  help='features for the last layer of affine')
    parser.add_argument('--output_ch',    type=int, default=3,                    help='number of output channels of the deformation field')
    
    
    # Train
    parser.add_argument('--seed',           type=int,   default=42,     help='random seed')
    parser.add_argument('--num_gpus',       type=int,   default=1,      help='number of GPUs available. Use 0 for CPU mode.')
    parser.add_argument('--start_ep',       type=int,   default=1,      help='start training epoch')
    parser.add_argument('--n_epochs',       type=int,   default=1000,   help='maximum training epochs')
    parser.add_argument('--batch_size',     type=int,   default=1,      help='batch size')
    parser.add_argument('--train_split',    type=float, default=0.8,    help='percentage to split the dataset for training')
    parser.add_argument('--alpha_value',    type=float, default=0.01,      help='parameter for the affine penalty loss')
    parser.add_argument('--beta_value',     type=float, default=0.1,      help='parameter for the deformation penalty loss') #0.0005
    parser.add_argument('--gamma_value',    type=float, default=0.1,    help='gamma parameter for the discriminator (feature matching loss: MSE)')
    parser.add_argument('--lr',             type=float, default=0.0001, help='learning rate')
    parser.add_argument('--beta1',          type=float, default=0.9,    help='adam optimizer beta1')
    parser.add_argument('--beta2',          type=float, default=0.999,  help='adam optimizer beta2')
    
    # Re-train
    parser.add_argument('--re_train',       type=bool,  default=False,  help='re-train a model')
    parser.add_argument('--chk_aff_to_load',type=str,   default='',     help='affine checkpoint to be load when we re-train the model')
    parser.add_argument('--chk_ela_to_load',type=str,   default='',     help='elastic checkpoint to be load when we re-train the model')
    parser.add_argument('--chk_disc_to_load',type=str,  default='',     help='discriminator checkpoint to be load when we re-train the model')
    
    args = parser.parse_args()
    
    main(args)
    