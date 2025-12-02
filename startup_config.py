import os
import torch
import numpy as np
import random

def set_random_seed(seed, args):
    """ 
    Fix random seed for reproduction
    """
    
    # set random seed for cpu
    random.seed(seed)
    
    # set random seed for numpy
    np.random.seed(seed)
    
    # set random seed for torch
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        # set random seed for cuda
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        if args.cudnn_deterministic_toggle:
            torch.backends.cudnn.deterministic = True
        if args.cudnn_benchmark_toggle:
            torch.backends.cudnn.benchmark = True