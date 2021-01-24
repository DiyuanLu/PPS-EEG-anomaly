import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
import pickle
import sys

from aae import AAE
from input_pipeline import csv_reader_dataset, get_train_val_files, get_data_files_LOO
from utils import get_run_logdir, plot_dict_loss

LOO = True # False
paths_platforms = {"Lu_laptop":{"PPS_data_path":"C:/Users/LDY/Desktop/EPG/EPG_data/data/3d/PPS",
                                "Ctrl_data_path": "C:/Users/LDY/Desktop/EPG/EPG_data/data/3d/control",
                                "root_logdir": "C:/Users/LDY/Desktop/EPG/EPG_data/results"
                                },
                   "FIAS_cluster": {"PPS_data_path": "/home/epilepsy-data/data/PPS-rats-from-Sebastian/PPS-Rats",
                                    "Ctrl_data_path": "/home/epilepsy-data/data/PPS-rats-from-Sebastian/Control-Rats",
                                    "root_logdir": "/home/epilepsy-data/data/PPS-rats-from-Sebastian/resultsl-7rats"
                                    },
                   "Farahat": {"PPS_data_path": '/home/farahat/Documents/data',
                               "Ctrl_data_path": '/home/farahat/Documents/data', # TODO: your control dir
                               "root_logdir": '/home/farahat/Documents/my_logs'
                               }
                   }

platform = "FIAS_cluster"
PPS_data_path = paths_platforms[platform]["PPS_data_path"]
Ctrl_data_path = paths_platforms[platform]["Ctrl_data_path"]
root_logdir = paths_platforms[platform]["root_logdir"]
#
n_sec_per_sample = 1
sampling_rate = 512
input_size = n_sec_per_sample * sampling_rate
h_dim = 512
z_dim = 16
n_epochs=100
batch_size = 128
LOO_animals = ["1275", "1276", "32140", "32141"]
n_files2use = 5


for LOO_animal in LOO_animals:
    # create output dir
    run_logdir = get_run_logdir(root_logdir, LOO_animal)

    f = open(os.path.join(run_logdir, 'log_file.out'), 'w')
    sys.stdout = f
    
    train_files, valid_files = [], []
    if_LOO_ctrl = True if "326" in LOO_animal else False  # determines whether it is in the LOO control rats case, then we need to get all PPS data and only do LOO in control
    if LOO:
        PPS_train_files, PPS_valid_files = get_data_files_LOO(PPS_data_path,
                                                              train_valid_split=True,
                                                              train_percentage=0.75, num2use=15,
                                                              LOO_ID=LOO_animal,
                                                              if_LOO_ctrl=if_LOO_ctrl,
                                                              current_folder="PPS", log_dir=run_logdir)
        Ctrl_train_files, Ctrl_valid_files = get_data_files_LOO(Ctrl_data_path,
                                                                train_valid_split=True,
                                                                train_percentage=0.75,
                                                                num2use=100,
                                                                LOO_ID=LOO_animal,
                                                                if_LOO_ctrl=if_LOO_ctrl,
                                                                current_folder="Ctrl", log_dir=run_logdir)
        train_files.extend(PPS_train_files)
        train_files.extend(Ctrl_train_files)
        valid_files.extend(PPS_valid_files)
        valid_files.extend(Ctrl_valid_files)
    else:
        train_files, valid_files = get_train_val_files(PPS_data_path, train_valid_split=True, train_percentage=0.8, num2use=n_files2use, log_dir=run_logdir)

    train_set = csv_reader_dataset(train_files, batch_size=batch_size, n_sec_per_sample=n_sec_per_sample,
                           sr=sampling_rate)
    valid_set = csv_reader_dataset(valid_files, batch_size=batch_size, n_sec_per_sample=n_sec_per_sample,
                           sr=sampling_rate)
    
    train_set = train_set.take(150)   # what does this do?
    
    # the model should be trained with the data from all rats at the same time. Not one after another.
    model = AAE(input_size, h_dim, z_dim, run_logdir)
    # model.print_trainable_weights_count()
    model.plot_models()
    metrics = model.train(n_epochs, train_set, valid_set)
    
    print("Save metrics")
    with open(run_logdir+'/metrics.pickle', 'wb') as handle:
        pickle.dump(metrics, handle)
    plot_dict_loss(metrics, run_logdir)
    
    print("Save model")
    model.save()
    model.clear_model()
    f.close()
    
    
    
    
    
