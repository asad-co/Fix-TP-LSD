# config/train_config.py
import os
from modeling.TP_Net import Res320, Res160
from modeling.Hourglass import HourglassNet

class TrainParameters:
    def __init__(self):
        # Dataset paths
        self.dataset_dir = '/content/dataset'
        self.train_file = os.path.join(self.dataset_dir, 'train.txt')
        self.val_file = os.path.join(self.dataset_dir, 'val.txt')
        self.image_dir = os.path.join(self.dataset_dir, 'images')
        self.label_dir = os.path.join(self.dataset_dir, 'labels')
        
        # Training settings
        self.save_path = 'log/train/'
        self.batch_size = 4
        self.num_workers = 4
        self.head = {'center': 1, 'dis': 4, 'line': 1}
        self.cuda = True
        self.epochs = 100
        self.save_interval = 5
        
        # Learning rate settings
        self.learning_rate = 1e-4
        self.lr_decay_epochs = [40, 60, 80]
        self.lr_decay_rate = 0.5
        
        # Loss weight settings
        self.center_weight = 1.0
        self.dis_weight = 0.5
        self.line_weight = 1.0
        
        # Optimizer weight settings
        self.optim_weight = {
            'back': 1.0,
            'center': 1.0,
            'dis': 1.0,
            'line': 1.0
        }
        
        # Model selection and loading
        self.resume = False
        self.selftrain = True
        
        # Choose one of the model architectures
        # self.model = Res320(self.head)
        # self.load_model_path = './pretraineds/Res320.pth'
        # self.inres = (320, 320)
        # self.outres = (320, 320)
        
        # TP-LSD-Lite
        self.model = Res160(self.head)
        self.load_model_path = './pretraineds/Res160.pth'
        self.inres = (320, 320)
        self.outres = (320, 320)
        
        # Enable logging
        self.logger = True
        
        os.makedirs(self.save_path, exist_ok=True)
        if self.cuda == False:
            raise Exception('CPU version for training is not implemented.')