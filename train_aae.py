import matplotlib
matplotlib.use('Agg')

import os
import pickle
import sys
import tensorflow as tf
import argparse

from aae import AAE
from input_pipeline import csv_reader_dataset, get_train_val_files, get_data_files_LOO, csv_reader_dataset_testtest, v2_create_dataset
from utils import get_run_logdir, plot_dict_loss, load_parameters, get_dirs_with_platform, copy_save_all_files
from scanning import scan_animals_with_pretrained_model

parser = argparse.ArgumentParser()
parser.add_argument(
    '--yaml_file', default="./parameters.yaml",
    help="Json file path for experiment parameters"
)
params = parser.parse_args()
args = load_parameters(params.yaml_file)

if not args.if_to_cluster:
    # data path related parameters
    args.pps_data_path, args.ctrl_data_path, args.root_logdir = get_dirs_with_platform(args.platform)
    args.run_logdir = get_run_logdir(args.root_logdir, args.LOO_animal, args)

if args.platform == "Farahat":
    tf.enable_eager_execution()

# save all files to experiemnt run_logdir
copy_save_all_files(args)

if not args.if_scanning:
    # for LOO_animal in args.LOO_animals:
    # create output dir when it is not from cluster queue

    f = open(os.path.join(args.run_logdir, 'log_file.out'), 'w')
    sys.stdout = f
    
    train_files, valid_files = [], []
    if_LOO_ctrl = True if "326" in args.LOO_animal else False  # determines whether it is in the LOO control rats case, then we need to get all pps data and only do LOO in control
    if args.LOO:
        pps_train_files, pps_valid_files = get_data_files_LOO(
            args.pps_data_path, args,
            train_valid_split=True,
            LOO_ID=args.LOO_animal,
            if_LOO_ctrl=if_LOO_ctrl,
            current_folder="pps")
        if args.if_include_ctrl:  # Give the choice of excluding ctrl rats
            ctrl_train_files, ctrl_valid_files = get_data_files_LOO(
                args.ctrl_data_path, args,
                train_valid_split=True,
                LOO_ID=args.LOO_animal,
                if_LOO_ctrl=if_LOO_ctrl,
                current_folder="ctrl")
            train_files.extend(ctrl_train_files)
            valid_files.extend(ctrl_valid_files)
        train_files.extend(pps_train_files)
        valid_files.extend(pps_valid_files)
    else:  # the following funtion is not working right
        train_files, valid_files = get_train_val_files(args.pps_data_path,
                                                       train_valid_split=True,
                                                       train_percentage=args.train_percentage,
                                                       num2use=args.n_files2use,
                                                       log_dir=args.run_logdir)
    
    train_set = v2_create_dataset(train_files, batch_size=args.batch_size,
                                   n_sec_per_sample=args.n_sec_per_sample,
                                   sr=args.sampling_rate)
    valid_set = v2_create_dataset(valid_files, batch_size=args.batch_size,
                                   n_sec_per_sample=args.n_sec_per_sample,
                                   sr=args.sampling_rate)
    
    # the model should be trained with the data from all rats at the same time. Not one after another.
    
    model = AAE(args.input_size, args.h_dim, args.z_dim, args.run_logdir)
    model.print_trainable_weights_count()
    # model.plot_models()
    metrics = model.train(args.n_epochs, train_set, valid_set)
    
    print("Save metrics")
    with open(args.run_logdir + '/metrics.pickle', 'wb') as handle:
        pickle.dump(metrics, handle)
    plot_dict_loss(metrics, args.run_logdir)
    
    print("Save model")
    model.save()
    model.clear_model()
    f.close()
else:
    scan_animals_with_pretrained_model(args)

    
        
        
        
    
    
