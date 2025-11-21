"""
Training script for HSCL
"""
from run import HSCL_run

HSCL_run(model_name='hscl', dataset_name='mosi', is_tune=False, seeds=[111], model_save_dir="./pt",
         res_save_dir="./result", log_dir="./log", mode='train', is_distill=True)
