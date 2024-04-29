
import os
import json
from   train import Train
'''from   train_elastic_net import TrainElastic
from   train_recursive_elastic_net import TrainRecursiveElastic
from   train_subtraction import TrainSubtractionElastic
from   train_concatenation import TrainConcatenationElastic'''


class Handler(object):
    def __init__(self, args):
        self.args       = args
        self.task       = args.task
        self.output_dir = args.output_dir
        self.save_config_file()
        
    def save_config_file(self):
        exp_dir = self.output_dir + self.task 
        if not os.path.isdir(exp_dir):
            os.makedirs(exp_dir)
            
        file_name = os.path.join(exp_dir, 'config.txt')
        with open(file_name, 'w') as f:
            json.dump(self.args.__dict__, f, indent=2)
                

    def train_affine(self):
        self.train = Train(args=self.args)
        self.train.run_train()
    
    
    '''def train_elastic(self):
        self.train = TrainElastic(args=self.args)
        self.train.run_train()
        
    def train_recursive_elastic(self):
        self.train = TrainRecursiveElastic(args=self.args)
        self.train.run_train()
        
    def train_subtraction_elastic(self):
        self.train = TrainSubtractionElastic(args=self.args)
        self.train.run_train()
        
    def train_concatenation_elastic(self):
        self.train = TrainConcatenationElastic(args=self.args)
        self.train.run_train()'''
        
    
    '''def test_registration_baseline(self):
        self.eval = Evaluation(args=self.args)
        self.eval.run_eval()'''
    
    
    def visualization(self):
        pass

    
    def run(self ):
        if self.task == 'train-plastic':#'train-affine':
            print("Training!!!!")
            self.train_affine()
        
        '''elif self.task == 'train-elastic':
            print("Training!!!!")
            self.train_elastic()
            
        elif self.task == 'train-recursive-elastic':
            print("Training!!!!")
            self.train_recursive_elastic()
        
        elif self.task == 'train-subtraction-elastic':
            print("Training!!!!")
            self.train_subtraction_elastic()
        
        elif self.task == 'train-concatenation-elastic': #train-concatenation-elastic
            print("Training!!!!")
            self.train_concatenation_elastic()
        
            
        elif self.task == 'test':
            print('Evaluation!!!!')
            #self.test_registration_baseline()
        elif self.task == 'vis':
            print('Under development...!')
        else:
            raise NotImplementedError('undefined task!')'''
