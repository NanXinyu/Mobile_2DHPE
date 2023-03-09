import _init_paths
from core.model import model_summary 
from config import cfg

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


if __name__ == '__main__':
    
    model_summary(cfg)