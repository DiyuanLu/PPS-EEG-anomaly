
from input_pipeline import csv_reader_dataset, get_train_val_files, get_data_files_LOO
from utils import get_run_logdir, plot_dict_loss
import numpy as np
from aae import AAE
import os
import pickle


LOO = True # False
data_path = "C:/Users/LDY/Desktop/EPG/EPG_data/data/3d/PPS"  #/home/epilepsy-data/data/PPS-rats-from-Sebastian/PPS-Rats" #'/home/farahat/Documents/data/'
root_logdir = "C:/Users/LDY/Desktop/EPG/EPG_data/results" #'/home/farahat/Documents/my_logs'
#"/home/epilepsy-data/data/PPS-rats-from-Sebastian/resultsl-7rats"
input_size = 2560
h_dim = 512
z_dim = 80
n_epochs=100
batch_size = 128
LOO_animals = ["1275", "1276", "32140", "32141"]

for LOO_animal in LOO_animals:
    if LOO:
        train_files, valid_files = get_data_files_LOO(data_path, train_valid_split=True,
                                                      train_percentage=0.75, num2use=10,
                                                       LOO_ID = LOO_animal)
    
    else:
        train_files, valid_files = get_train_val_files(data_path)
    # here always have valid_set, so the output of previous function should always have train and valid lists
    train_set = csv_reader_dataset(train_files, batch_size=batch_size)
    valid_set = csv_reader_dataset(valid_files, batch_size=batch_size)
    
    train_set = train_set.take(150)
    run_logdir = get_run_logdir(root_logdir, LOO_animal)
    
    # the model should be trained with the data from all rats at the same time. Not one after another.
    
    model = AAE(input_size, h_dim, z_dim, run_logdir)
    model.print_trainable_weights_count()
    model.plot_models()
    metrics = model.train(n_epochs, train_set, valid_set)
    with open(run_logdir+'/metrics.pickle', 'wb') as handle:
        pickle.dump(metrics, handle)
    plot_dict_loss(metrics, run_logdir)
    model.save()
    
    model.clear_model()
    
    
    
    
    
