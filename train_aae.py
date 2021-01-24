import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
import pickle
import sys

from aae import AAE
from input_pipeline import csv_reader_dataset, get_train_val_files, get_data_files_LOO
from utils import get_run_logdir, plot_dict_loss, Struct


paths_platforms = {"Lu_laptop":{"pps_data_path":"C:/Users/LDY/Desktop/EPG/EPG_data/data/3d/PPS",
                                "ctrl_data_path": "C:/Users/LDY/Desktop/EPG/EPG_data/data/3d/control",
                                "root_logdir": "C:/Users/LDY/Desktop/EPG/EPG_data/results"
                                },
                   "FIAS_cluster": {"pps_data_path": "/home/epilepsy-data/data/PPS-rats-from-Sebastian/PPS-Rats",
                                    "ctrl_data_path": "/home/epilepsy-data/data/PPS-rats-from-Sebastian/Control-Rats",
                                    "root_logdir": "/home/epilepsy-data/data/PPS-rats-from-Sebastian/resultsl-7rats"
                                    },
                   "Farahat": {"pps_data_path": '/home/farahat/Documents/data',
                               "ctrl_data_path": '/home/farahat/Documents/data', # TODO: your control dir
                               "root_logdir": '/home/farahat/Documents/my_logs'
                               }
                   }

platform = "FIAS_cluster"
pps_data_path = paths_platforms[platform]["pps_data_path"]
ctrl_data_path = paths_platforms[platform]["ctrl_data_path"]
root_logdir = paths_platforms[platform]["root_logdir"]
#
parameters = {
    "LOO": True, # False
    "n_sec_per_sample": 1,
    "sampling_rate": 512,
    "input_size" : 512,
    "h_dim": 512,
    "z_dim": 16,
    "n_epochs": 100,
    "batch_size": 128,
    "LOO_animals": ["1275", "1276", "32140", "32141"], #["1275", "1276", "32140", "32141"],
    "n_pps2use": 20,  # 20,
    "n_ctrl2use": 100,  # 100,
    "train_percentage": 0.9,
    "pps_animals": ["1227", "1275", "1270", "1275", "1276", "32140", "32141"],#
    "ctrl_animals": ["3263", "3266", "3267"],
    "file_pattern": "new.csv",
    "if_include_ctrl": False # whether to include ctrl animals
}

args = Struct(**parameters)
assert args.input_size == args.n_sec_per_sample * args.sampling_rate, "input size is wrong!"

for LOO_animal in args.LOO_animals:
    # create output dir
    args.run_logdir = get_run_logdir(root_logdir, LOO_animal)

    f = open(os.path.join(args.run_logdir, 'log_file.out'), 'w')
    sys.stdout = f
    
    train_files, valid_files = [], []
    if_LOO_ctrl = True if "326" in LOO_animal else False  # determines whether it is in the LOO control rats case, then we need to get all pps data and only do LOO in control
    if args.LOO:
        pps_train_files, pps_valid_files = get_data_files_LOO(pps_data_path, args,
                                                              train_valid_split=True,
                                                              LOO_ID=LOO_animal,
                                                              if_LOO_ctrl=if_LOO_ctrl,
                                                              current_folder="pps")
        if args.if_include_ctrl:
            ctrl_train_files, ctrl_valid_files = get_data_files_LOO(ctrl_data_path,
                                                                    train_valid_split=True,
                                                                    LOO_ID=LOO_animal,
                                                                    if_LOO_ctrl=if_LOO_ctrl,
                                                                    current_folder="ctrl")
            train_files.extend(ctrl_train_files)
            valid_files.extend(ctrl_valid_files)
        train_files.extend(pps_train_files)
        valid_files.extend(pps_valid_files)
    else:
        train_files, valid_files = get_train_val_files(pps_data_path, train_valid_split=True, train_percentage=0.8, num2use=args.n_files2use, log_dir=args.run_logdir)

    train_set = csv_reader_dataset(train_files, batch_size=args.batch_size, n_sec_per_sample=args.n_sec_per_sample,
                           sr=args.sampling_rate)
    valid_set = csv_reader_dataset(valid_files, batch_size=args.batch_size, n_sec_per_sample=args.n_sec_per_sample,
                           sr=args.sampling_rate)
    
    train_set = train_set.take(150)   # what does this do?
    
    # the model should be trained with the data from all rats at the same time. Not one after another.
    model = AAE(args.input_size, args.h_dim, args.z_dim, args.run_logdir)
    # model.print_trainable_weights_count()
    # model.plot_models()
    metrics = model.train(args.n_epochs, train_set, valid_set)
    
    print("Save metrics")
    with open(args.run_logdir+'/metrics.pickle', 'wb') as handle:
        pickle.dump(metrics, handle)
    plot_dict_loss(metrics, args.run_logdir)
    
    print("Save model")
    model.save()
    model.clear_model()
    f.close()
    
    
    
    
    
