import nni
import os
import numpy as np
from utils import *
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the model with specified parameters")
    parser.add_argument('--dataset1', type=str, required=True, help='The first dataset name')
    parser.add_argument('--dataset2', type=str, required=True, help='The second dataset name')
    parser.add_argument('--entity', type=int, required=True, help='Entity number')
    parser.add_argument('--data_len', type=int, required=True, help='Length of data')
    parser.add_argument('--create_len', type=int, required=True, help='Length of create')
    parser.add_argument('--cycle', type=int, required=True, help='Length of cycle')
    return parser.parse_args()

def main():
    # Parse arguments from the terminal
    args = parse_arguments()
    
    params = nni.get_next_parameter()
    model_name = params["model"]["_name"]

    run_model(model_name, params["model"], args.dataset1, args.dataset2, args.entity)
    DbAN = evaluate_model(model_name, params["model"], args.dataset1, args.dataset2, args.entity, args.data_len, args.create_len, args.cycle)
    nni.report_final_result(DbAN)

if __name__ == "__main__":
    main()


