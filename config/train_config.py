# config/train_config.py
import os
# Don't import models directly here - defer initialization

class TrainParameters:
    def __init__(self):
        # Dataset paths
        self.dataset_dir = '/content/dataset'
        self.train_file = os.path.join(self.dataset_dir, 'train.txt')
        self.val_file = os.path.join(self.dataset_dir, 'val.txt')
        self.image_dir = os.path.join(self.dataset_dir, 'images')
        self.label_dir = os.path.join(self.dataset_dir, 'labels')
        
        # Training settings
        self.save_path = '/content/log/train/'
        self.batch_size = 4
        self.num_workers = 4
        self.head = {'center': 1, 'dis': 4, 'line': 1}
        self.cuda = True
        self.epochs = 10000
        self.save_interval = 200
        
        # Learning rate settings
        self.learning_rate = 1e-5
        self.lr_decay_epochs = [40, 60, 80]
        self.lr_decay_rate = 0.5
        
        # Loss weight settings
        self.center_weight = 0.01
        self.dis_weight = 0.01
        self.line_weight = 0.01
        
        # Optimizer weight settings
        self.optim_weight = {
            'back': 1.0,
            'center': 1.0,
            'dis': 1.0,
            'line': 1.0
        }
        
        # Model selection and loading
        self.resume = False
        self.selftrain = False
        
        # Model parameters without actual initialization
        self.model_type = 'Res512'  # Options: 'Res320', 'Res160', 'HourglassNet'
        self.load_model_path = './pretraineds/Res512.pth'
        self.inres = (512, 512)
        self.outres = (512, 512)
        
        # Enable logging
        self.logger = True
        
        os.makedirs(self.save_path, exist_ok=True)
        if self.cuda == False:
            raise Exception('CPU version for training is not implemented.')
        
    def create_model(self):
        """
        Create model instance on demand rather than at import time
        """
        # Import here to avoid premature CUDA initialization
        from modeling.TP_Net import Res320, Res160
        from modeling.Hourglass import HourglassNet
        
        if self.model_type == 'Res320':
            return Res320(self.head)
        elif self.model_type == 'Res160':
            return Res160(self.head)
        elif self.model_type == 'HourglassNet':
            return HourglassNet(self.head)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")