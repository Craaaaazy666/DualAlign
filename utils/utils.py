import yaml
import logging
import torch
from torch import nn
from torch.nn import functional as F

def check_cuda():
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print("CUDA avail., total {} CUDA devices:".format(device_count))
        for i in range(device_count):
            device = torch.device("cuda:{}".format(i))
            print("CUDA devices {}: {}".format(i, torch.cuda.get_device_name(i)))
            
            # get current device memory
            total_memory = torch.cuda.get_device_properties(device).total_memory
            allocated_memory = torch.cuda.memory_allocated(device)  
            reserved_memory = torch.cuda.memory_reserved(device)    
            free_memory = total_memory - allocated_memory - reserved_memory  
            print("  - Total memory: {:.2f} GB".format(total_memory / (1024 ** 3)))
            print("  - Allocated: {:.2f} GB".format(allocated_memory / (1024 ** 3)))
            print("  - Reserved: {:.2f} GB".format(reserved_memory / (1024 ** 3)))
            print("  - Remaining: {:.2f} GB".format(free_memory / (1024 ** 3)))
    else:
        print("CUDA unavail.")


def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup as many loggers as you want"""
    handler = logging.FileHandler(log_file, mode='w')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def masked_loss(out, label, mask):
    loss = F.cross_entropy(out, label, reduction='none')
    mask = mask.float()
    mask = mask / mask.mean()
    loss *= mask
    loss = loss.mean()
    return loss


def masked_acc(out, label, mask):
    # [node, f]
    pred = out.argmax(dim=1)
    correct = torch.eq(pred, label).float()
    mask = mask.float()
    mask = mask / mask.mean()
    correct *= mask
    acc = correct.mean()
    return acc


def sparse_dropout(x, rate, noise_shape):
    """
    :param x:
    :param rate:
    :param noise_shape: int scalar
    :return:
    """
    random_tensor = 1 - rate
    random_tensor += torch.rand(noise_shape).to(x.device)
    dropout_mask = torch.floor(random_tensor).byte()
    i = x._indices() # [2, 49216]
    v = x._values() # [49216]

    # [2, 4926] => [49216, 2] => [remained node, 2] => [2, remained node]
    i = i[:, dropout_mask]
    v = v[dropout_mask]

    out = torch.sparse.FloatTensor(i, v, x.shape).to(x.device)

    out = out * (1./ (1-rate))

    return out

def dot(x, y, sparse=False):
    if sparse:
        res = torch.sparse.mm(x, y)
    else:
        res = torch.mm(x, y)

    return res

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config