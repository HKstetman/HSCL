"""
Testing script for HSCL
"""
from run import HSCL_run

HSCL_run(model_name='hscl', dataset_name='mosei', is_tune=False, seeds=[1111], model_save_dir="./pt",
         res_save_dir="./result", log_dir="./log", mode='test', is_distill=False)
