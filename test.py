# import argparse
# import datetime
# import models
# import os
# import shutil
# import time
# import numpy as np
import torch
import torch.nn as nn
# import torch.backends.cudnn as cudnn
# from config import cfg, process_args
# from data import fetch_dataset, make_data_loader, separate_dataset_su, make_batchnorm_stats, make_batchnorm_dataset_su
# from metrics import Metric
# from utils import save, to_device, process_control, process_dataset, create_optimizer, create_scheduler, resume, collate
# from logger import Logger

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# cudnn.benchmark = True
# parser = argparse.ArgumentParser(description='cfg')
# for k in cfg:
#     exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
# parser.add_argument('--control_name', default=None, type=str)
# args = vars(parser.parse_args())
# process_args(args)


# if __name__ == "__main__":
import numpy as np
from scipy.stats import norm

x = torch.randn(1, 3, 32, 32)
print(x)
threshold = 0.1
x = torch.where(x >= threshold, x, torch.tensor(threshold))

print(x)


data = x.view(-1).detach().cpu().numpy()

pdf_values = norm.pdf(data)
print(pdf_values, len(pdf_values))
# To calculate the CDF, use the `cdf` function
cdf_values = norm.cdf(data)
print(cdf_values)
# # Calculate histogram
# hist, bin_edges = np.histogram(data, bins=100, density=True)

# # Calculate PDF
# pdf = hist / sum(hist)

# # Calculate CDF
# cdf = np.cumsum(pdf)

# print("PDF: ", pdf)
# print("CDF: ", cdf)