from input_pipeline import csv_reader_dataset, get_all_data_files, get_data_files_from_folder
from utils import get_run_logdir
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import itertools
import os
import scipy
import math
import seaborn as sns
sns.set(style="darkgrid")
import pandas as pd
from sklearn.decomposition import PCA


def scan_animals_with_pretrained_model(args):
    """
    Scan the whole data with a pretrained model
    :param args:
    :return:
    """
    data_path = '/home/farahat/Documents/data/'
    root_logdir = '/home/farahat/Documents/my_logs/final7/'
    # root_logdir = '/home/farahat/Documents/my_logs/'
    batch_size = 512
    # models = sorted([f for f in os.listdir(root_logdir)])
    
    # models = ['run_2020_01_27-12_38_39_1227']
    z_dim = 80
    
    for model_name in args.models[:]:
        print('working on: '+model_name)
        animal = os.path.basename(model_name).split("_")[-1]
        # animal = model_name[24:]
        if "326" in animal:
            data_path = args.ctrl_data_path
        else:
            data_path = args.pps_data_path
        animal_path = os.path.join(data_path, animal, animal)
    
        run_logdir = args.root_logdir + model_name
        output_directory = run_logdir +  '/stats/'
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)

        if args.LOO:
            epg_files = get_data_files_from_folder(animal_path+'/EPG/', train_valid_split=False)
            valid_files = get_data_files_from_folder(animal_path+'/BL/', train_valid_split=False)
            train_files = get_all_data_files(data_path, animal, train_valid_split=False)
        else:
            epg_files = get_data_files_from_folder(animal_path+'/EPG/', train_valid_split=False)
            train_files, valid_files = get_data_files_from_folder(animal_path+'/BL/')
        
        ###
        epg_set = csv_reader_dataset(epg_files, batch_size=batch_size, shuffle=False)
        valid_set = csv_reader_dataset(valid_files, batch_size=batch_size, shuffle=False)
        train_set = csv_reader_dataset(train_files, batch_size=batch_size, shuffle=False)
    
        encoder = tf.keras.models.load_model(run_logdir+'/encoder.h5')
        decoder = tf.keras.models.load_model(run_logdir+'/decoder.h5')
        disc_x = tf.keras.models.load_model(run_logdir+'/discriminator_x.h5')
    
    
        def compute_batch_distance(z):
            distance = []
            for i in range(z.shape[0]):
                distance.append(scipy.spatial.distance.euclidean(z[i].numpy(),np.zeros(z_dim)))
            return np.array(distance)
    
    
        def compute_distros(dataset, directory):
            if not os.path.exists(directory):
                os.mkdir(directory)
            errors = np.array([])
            probilities = np.array([])
            distances = np.array([])
            z_all = np.zeros(z_dim)
    
            for i, batch in enumerate(dataset):
                z = encoder(batch)
                z_all = np.vstack((z_all,z[:,:,0].numpy()))
    
                x_hat = decoder(z)
                prob = scipy.special.expit(disc_x(x_hat)[0]).ravel()
                probilities = np.concatenate((probilities,prob),axis=0)
    
    
                loss = np.square(batch-x_hat)[:,:,0]
                # error = loss.reshape(loss.shape[0]*loss.shape[1])
                error = np.mean(loss, axis=1).ravel()
                errors = np.concatenate((errors,error),axis=0)
    
                distance = compute_batch_distance(z[:,:,0])
                distances = np.concatenate((distances,distance),axis=0)
    
                if (i+1) % 10 == 0:
                    print('finished: '+str(i)+' batches')
            np.save(directory+'/errors.npy', errors)
            np.save(directory+'/probilities.npy', probilities)
            np.save(directory+'/distances.npy', distances)
            np.save(directory+'/z.npy', z_all[1:,:])
    
        compute_distros(train_set, output_directory+'train_data')
        compute_distros(valid_set, output_directory+'valid_data')
        compute_distros(epg_set, output_directory+'epg_data')
    
        ####################################################################
    
        t_errors = np.load(output_directory+'train_data'+'/errors.npy')
        t_probilities = np.load(output_directory+'train_data'+'/probilities.npy')
        t_distances = np.load(output_directory+'train_data'+'/distances.npy')
        t_z_all = np.load(output_directory+'train_data'+'/z.npy')
    
        v_errors = np.load(output_directory+'valid_data'+'/errors.npy')
        v_probilities = np.load(output_directory+'valid_data'+'/probilities.npy')
        v_distances = np.load(output_directory+'valid_data'+'/distances.npy')
        v_z_all = np.load(output_directory+'valid_data'+'/z.npy')
    
        e_errors = np.load(output_directory+'epg_data'+'/errors.npy')
        e_probilities = np.load(output_directory+'epg_data'+'/probilities.npy')
        e_distances = np.load(output_directory+'epg_data'+'/distances.npy')
        e_z_all = np.load(output_directory+'epg_data'+'/z.npy')
    
        fig = plt.figure(figsize=(10,10))
        sns.distplot(e_errors, kde=False, norm_hist=True, label='epg errors')
        sns.distplot(t_errors, kde=False, norm_hist=True, label='train errors')
        sns.distplot(v_errors, kde=False, norm_hist=True, label='valid errors')
        plt.yscale('log')
        plt.legend()
        plt.savefig(output_directory+'errors.png')
        plt.close()
    
        fig = plt.figure(figsize=(10,10))
        sns.distplot(e_probilities, kde=False, norm_hist=True, label='epg probabilities')
        sns.distplot(t_probilities, kde=False, norm_hist=True, label='train probabilities')
        sns.distplot(v_probilities, kde=False, norm_hist=True, label='valid probabilities')
        plt.legend()
        plt.savefig(output_directory+'probabilities.png')
        plt.close()
    
        fig = plt.figure(figsize=(10,10))
        sns.distplot(e_distances, kde=False, norm_hist=True, label='epg distances')
        sns.distplot(t_distances, kde=False, norm_hist=True, label='train distances')
        sns.distplot(v_distances, kde=False, norm_hist=True, label='valid distances')
        plt.legend()
        plt.savefig(output_directory+'distances.png')
        plt.close()
    
        # reshaped_t_errors = np.reshape(t_errors, (int(t_errors.shape[0]/2560),2560))
        # whole_segment_t_errors = np.mean(reshaped_t_errors, axis = 1)
        whole_segment_t_errors = t_errors
        np.save(output_directory+'train_data'+'/whole_segment_t_errors.npy', whole_segment_t_errors)
        # whole_segment_t_errors = np.load(output_directory+'train_data'+'/whole_segment_t_errors.npy')
        moving_average = pd.Series(whole_segment_t_errors).rolling(720*12).mean()
        moving_std = pd.Series(whole_segment_t_errors).rolling(720*12).std()
        fig = plt.figure(figsize=(20,10))
        plt.scatter(np.arange(int(whole_segment_t_errors.shape[0])),whole_segment_t_errors, color='orange', label='errors', marker='.', s=1)
        plt.plot(moving_average, linewidth=2, color='black', label='moving average')
        plt.fill_between(moving_std.index, (moving_average-moving_std), (moving_average+moving_std), color='red', alpha=.2, label=' moving std')
        # plt.ylim([0,2000])
        plt.xlabel('Time in days')
        plt.ylabel('Reconstruction error')
        ticks = np.arange(0,whole_segment_t_errors.shape[0], 720*24)
        plt.xticks(ticks, np.arange(0,len(ticks), 1))
        plt.legend()
        plt.savefig(output_directory+'t_whole_segment_errors.png')
        plt.close()
    
        th90 = np.percentile(whole_segment_t_errors, 90)
        th95 = np.percentile(whole_segment_t_errors, 95)
        th99 = np.percentile(whole_segment_t_errors, 99)
    
        # reshaped_v_errors = np.reshape(v_errors, (int(v_errors.shape[0]/2560),2560))
        # whole_segment_v_errors = np.mean(reshaped_v_errors, axis = 1)
        whole_segment_v_errors = v_errors
        np.save(output_directory+'valid_data'+'/whole_segment_v_errors.npy', whole_segment_v_errors)
        # whole_segment_v_errors = np.load(output_directory+'valid_data'+'/whole_segment_v_errors.npy')
        moving_average = pd.Series(whole_segment_v_errors).rolling(720).mean()
        moving_std = pd.Series(whole_segment_v_errors).rolling(720).std()
        fig = plt.figure(figsize=(20,10))
        plt.scatter(np.arange(int(whole_segment_v_errors.shape[0])),whole_segment_v_errors, color='orange', label='errors', marker='.', s=1)
        plt.plot(moving_average, linewidth=2, color='black', label='moving average')
        plt.fill_between(moving_std.index, (moving_average-moving_std), (moving_average+moving_std), color='red', alpha=.2, label=' moving std')
        plt.hlines([th90, th95, th99], 0, len(whole_segment_v_errors), colors='green', linewidth=2, linestyles='dashed', label='90th, 95th, 99th percentiles')
        # plt.ylim([0,2000])
        plt.xlabel('Time in hours')
        plt.ylabel('Reconstruction error')
        ticks = np.arange(0,whole_segment_v_errors.shape[0], 720)
        plt.xticks(ticks, np.arange(0,len(ticks), 1))
        plt.legend()
        plt.savefig(output_directory+'v_whole_segment_errors.png')
        plt.close()
    
        # reshaped_e_errors = np.reshape(e_errors, (int(e_errors.shape[0]/2560),2560))
        # whole_segment_e_errors = np.mean(reshaped_e_errors, axis = 1)
        whole_segment_e_errors = e_errors
        np.save(output_directory+'epg_data'+'/whole_segment_e_errors.npy', whole_segment_e_errors)
        # whole_segment_e_errors = np.load(output_directory+'epg_data'+'/whole_segment_e_errors.npy')
        moving_average = pd.Series(whole_segment_e_errors).rolling(720*12).mean()
        moving_std = pd.Series(whole_segment_e_errors).rolling(720*12).std()
        fig = plt.figure(figsize=(20,10))
        plt.scatter(np.arange(int(whole_segment_e_errors.shape[0])),whole_segment_e_errors, color='orange', label='errors', marker='.', s=1)
        plt.plot(moving_average, linewidth=2, color='black', label='moving average')
        plt.fill_between(moving_std.index, (moving_average-moving_std), (moving_average+moving_std), color='red', alpha=.2, label=' moving std')
        plt.hlines([th90, th95, th99], 0, len(whole_segment_e_errors), colors='green', linewidth=2, linestyles='dashed', label='90th, 95th, 99th percentiles')
        # plt.ylim([0,2000])
        plt.xlabel('Time in days')
        plt.ylabel('Reconstruction error')
        ticks = np.arange(0,whole_segment_e_errors.shape[0], 720*24)
        plt.xticks(ticks, np.arange(0,len(ticks)*24, 1))
        plt.legend()
        plt.savefig(output_directory+'e_whole_segment_errors.png')
        plt.close()
    
        fig = plt.figure(figsize=(10,10))
        sns.distplot(whole_segment_e_errors, kde=False, norm_hist=True, label='epg errors')
        sns.distplot(whole_segment_t_errors, kde=False, norm_hist=True, label='train errors')
        sns.distplot(whole_segment_v_errors, kde=False, norm_hist=True, label='valid errors')
        plt.legend()
        plt.savefig(output_directory+'whole_segment_errors.png')
        plt.close()
    
        th_whole_segment_t_errors = np.copy(whole_segment_t_errors)
        th_whole_segment_t_errors[th_whole_segment_t_errors < th99] = 0
        fig = plt.figure(figsize=(20,10))
        plt.scatter(np.arange(len(th_whole_segment_t_errors)),th_whole_segment_t_errors, marker='.', color='orange', s=1)
        plt.hlines(th99, 0, len(whole_segment_t_errors), colors='green', linewidth=1, linestyles='dashed', label='99th percentiles')
        # plt.ylim([0,2000])
        plt.xlabel('Time in days')
        plt.ylabel('Reconstruction error')
        ticks = np.arange(0,whole_segment_t_errors.shape[0], 720*24)
        plt.xticks(ticks, np.arange(0,len(ticks)*24, 1))
        plt.savefig(output_directory+'th_t_whole_segment_errors.png')
        plt.close()
    
        th_whole_segment_v_errors = np.copy(whole_segment_v_errors)
        th_whole_segment_v_errors[th_whole_segment_v_errors < th99] = 0
        fig = plt.figure(figsize=(20,10))
        plt.scatter(np.arange(len(th_whole_segment_v_errors)),th_whole_segment_v_errors, marker='.', color='orange', s=1)
        plt.hlines(th99, 0, len(whole_segment_v_errors), colors='green', linewidth=1, linestyles='dashed', label='99th percentiles')
        # plt.ylim([0,2000])
        plt.xlabel('Time in days')
        plt.ylabel('Reconstruction error')
        ticks = np.arange(0,whole_segment_v_errors.shape[0], 720*24)
        plt.xticks(ticks, np.arange(0,len(ticks)*24, 1))
        plt.savefig(output_directory+'th_v_whole_segment_errors.png')
        plt.close()
    
        th_whole_segment_e_errors = np.copy(whole_segment_e_errors)
        th_whole_segment_e_errors[th_whole_segment_e_errors < th99] = 0
        fig = plt.figure(figsize=(20,10))
        plt.scatter(np.arange(len(th_whole_segment_e_errors)),th_whole_segment_e_errors, marker='.', color='orange', s=1)
        plt.hlines(th99, 0, len(whole_segment_e_errors), colors='green', linewidth=1, linestyles='dashed', label='99th percentiles')
        # plt.ylim([0,2000])
        plt.xlabel('Time in days')
        plt.ylabel('Reconstruction error')
        ticks = np.arange(0,whole_segment_e_errors.shape[0], 720*24)
        plt.xticks(ticks, np.arange(0,len(ticks)*24, 1))
        plt.savefig(output_directory+'th_e_whole_segment_errors.png')
        plt.close()
    
    
    
        th90_d = np.percentile(t_distances, 90)
        th95_d = np.percentile(t_distances, 95)
        th99_d = np.percentile(t_distances, 99)
    
        moving_average = pd.Series(t_distances).rolling(720*12).mean()
        moving_std = pd.Series(t_distances).rolling(720*12).std()
        fig = plt.figure(figsize=(20,10))
        plt.scatter(np.arange(int(t_distances.shape[0])),t_distances, color='orange', label='errors', marker='.', s=1)
        plt.plot(moving_average, linewidth=2, color='black', label='moving average')
        plt.fill_between(moving_std.index, (moving_average-moving_std), (moving_average+moving_std), color='red', alpha=.2, label=' moving std')
        # plt.ylim([0,4])
        plt.xlabel('Time in days')
        plt.ylabel('Reconstruction error')
        ticks = np.arange(0,t_distances.shape[0], 720*24)
        plt.xticks(ticks, np.arange(0,len(ticks), 1))
        plt.legend()
        plt.savefig(output_directory+'t_distances.png')
        plt.close()
    
        moving_average = pd.Series(v_distances).rolling(720).mean()
        moving_std = pd.Series(v_distances).rolling(720).std()
        fig = plt.figure(figsize=(20,10))
        plt.scatter(np.arange(int(v_distances.shape[0])),v_distances, color='orange', label='errors', marker='.', s=1)
        plt.plot(moving_average, linewidth=2, color='black', label='moving average')
        plt.fill_between(moving_std.index, (moving_average-moving_std), (moving_average+moving_std), color='red', alpha=.2, label=' moving std')
        plt.hlines([th90_d, th95_d, th99_d], 0, len(v_distances), colors='green', linewidth=2, linestyles='dashed', label='90th, 95th, 99th percentiles')
        # plt.ylim([0,4])
        plt.xlabel('Time in hours')
        plt.ylabel('Reconstruction error')
        ticks = np.arange(0,v_distances.shape[0], 720)
        plt.xticks(ticks, np.arange(0,len(ticks), 1))
        plt.legend()
        plt.savefig(output_directory+'v_distances.png')
        plt.close()
    
        moving_average = pd.Series(e_distances).rolling(720*12).mean()
        moving_std = pd.Series(e_distances).rolling(720*12).std()
        fig = plt.figure(figsize=(20,10))
        plt.scatter(np.arange(int(e_distances.shape[0])),e_distances, color='orange', label='errors', marker='.', s=1)
        plt.plot(moving_average, linewidth=2, color='black', label='moving average')
        plt.fill_between(moving_std.index, (moving_average-moving_std), (moving_average+moving_std), color='red', alpha=.2, label=' moving std')
        plt.hlines([th90_d, th95_d, th99_d], 0, len(e_distances), colors='green', linewidth=2, linestyles='dashed', label='90th, 95th, 99th percentiles')
        # plt.ylim([0,4])
        plt.xlabel('Time in days')
        plt.ylabel('Reconstruction error')
        ticks = np.arange(0,e_distances.shape[0], 720*24)
        plt.xticks(ticks, np.arange(0,len(ticks)*24, 1))
        plt.legend()
        plt.savefig(output_directory+'e_distances.png')
        plt.close()
    
    
        # pca = PCA(n_components=2)
        # pca.fit(e_z_all)
        # transformed_e = pca.transform(e_z_all)
        # fig = plt.figure(figsize=(20,20))
        # plt.scatter(transformed_e[:, 0], transformed_e[:,1], cmap='jet', c=whole_segment_e_errors, s=2)
        # plt.xlabel('PC1')
        # plt.ylabel('PC2')
        # plt.legend()
        # plt.colorbar()
        # plt.show()
        # plt.savefig(output_directory+'pca_z_e_color.png')
        # plt.close()
    
    
        test_window_in_minutes = [1,5,10,15,30,60]
        for window_in_minutes in test_window_in_minutes:
            window = int((window_in_minutes*60)/5)
            # r = np.reshape(th_whole_segment_t_errors[:len(th_whole_segment_t_errors)//window*window], (-1, window))
            # frequency_t = np.sum(np.where(r>0, 1, 0),axis=1)
            r = np.where(whole_segment_t_errors>th99, 1, 0)
            frequency_t = pd.Series(r).rolling(window).sum()
            fig = plt.figure(figsize=(20,10))
            plt.plot(np.arange(len(frequency_t)),frequency_t)
            plt.xlabel('Time in days')
            plt.ylabel('#suprathrehold segments per '+str(window_in_minutes)+' minutes')
            ticks = np.arange(0,frequency_t.shape[0], (720)*24)
            plt.xticks(ticks, np.arange(0,len(ticks)*24, 1))
            plt.savefig(output_directory+'frequency_t_'+str(window_in_minutes)+' minutes_no.png')
            plt.close()
    
            th99_frequency = np.nanpercentile(frequency_t, 99)
    
            # r = np.reshape(th_whole_segment_v_errors[:len(th_whole_segment_v_errors)//window*window], (-1, window))
            # frequency_v = np.sum(np.where(r>0, 1, 0),axis=1)
            r = np.where(whole_segment_v_errors>th99, 1, 0)
            frequency_v = pd.Series(r).rolling(window).sum()
            fig = plt.figure(figsize=(20,10))
            plt.plot(np.arange(len(frequency_v)),frequency_v)
            plt.axhline(th99_frequency, c='g', linewidth=3, linestyle='dotted')
            if np.any(frequency_v.values > th99_frequency):
                plt.axvline(np.where(frequency_v > th99_frequency)[0][0], c='r', linewidth=3, linestyle='dashed')
            plt.xlabel('Time in days')
            plt.ylabel('#suprathrehold segments per '+str(window_in_minutes)+' minutes')
            ticks = np.arange(0,frequency_v.shape[0], (720)*24)
            plt.xticks(ticks, np.arange(0,len(ticks)*24, 1))
            plt.savefig(output_directory+'frequency_v_'+str(window_in_minutes)+' minutes_no.png')
            plt.close()
    
            # r = np.reshape(th_whole_segment_e_errors[:len(th_whole_segment_e_errors)//window*window], (-1, window))
            # frequency_e = np.sum(np.where(r>0, 1, 0),axis=1)
            r = np.where(whole_segment_e_errors>th99, 1, 0)
            frequency_e = pd.Series(r).rolling(window).sum()
            fig = plt.figure(figsize=(20,10))
            plt.plot(np.arange(len(frequency_e)),frequency_e)
            plt.axhline(th99_frequency, c='g', linewidth=3, linestyle='dotted')
            if np.any(frequency_e.values > th99_frequency):
                plt.axvline(np.where(frequency_e > th99_frequency)[0][0], c='r', linewidth=3, linestyle='dashed')
            plt.xlabel('Time in days')
            plt.ylabel('#suprathrehold segments per'+str(window_in_minutes)+' minutes')
            ticks = np.arange(0,frequency_e.shape[0], (720)*24)
            plt.xticks(ticks, np.arange(0,len(ticks)*24, 1))
            plt.savefig(output_directory+'frequency_e_'+str(window_in_minutes)+' minutes_no.png')
            plt.close()
    
        #############################################################################
