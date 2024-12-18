import os
import gc
import scipy
import ipdb
import math
import numpy as np
import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
sns.set(style="darkgrid")
import pandas as pd
from tqdm import tqdm

from input_pipeline import csv_reader_dataset, get_data_files_from_folder, get_all_data_files, v2_create_dataset
from utils import get_run_logdir, set_gpu_memory_growth
# tf.enable_eager_execution()
from sklearn.decomposition import PCA
# set_gpu_memory_growth()

random_seed = 42
tf.random.set_seed(random_seed)
np.random.seed(random_seed)

precomputed = False
LOO = True
data_path_general = "/home/epilepsy-data/data/PPS-rats-from-Sebastian"
root_logdir = '/home/epilepsy-data/data/PPS-rats-from-Sebastian/amr_logs/final_128_0.1'
save_root = '/home/epilepsy-data/data/PPS-rats-from-Sebastian/results-7rats/final_128_0.1'
batch_size = 1024
models = sorted([f for f in os.listdir(root_logdir) if "run_EPG_anomaly" in f])
z_dim = 128

for model_name in models[:]:
    print('working on: ' + model_name)

    # animal = model_name[40:]
    animal = os.path.basename(model_name).split("_")[-1]
    if animal == "1275":
        continue
    # animal_path = data_path+animal
    if "326" in animal:
        data_path = os.path.join(data_path_general, 'Control-Rats')
    else:
        data_path = os.path.join(data_path_general,'PPS-Rats')
    animal_path = os.path.join(data_path, animal, animal)

    output_logdir = os.path.join(save_root, model_name)
    run_logdir = os.path.join(root_logdir,  model_name)
    
    if not os.path.exists(output_logdir):
        os.mkdir(output_logdir)
    output_directory = os.path.join(output_logdir, 'stats_{}'.format(animal))
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
    for subdir in ["train_data", "epg_data", "valid_data"]:
        if not os.path.exists(os.path.join(output_directory, subdir)):
            os.mkdir(os.path.join(output_directory, subdir))
            print("Made path: {}".format(os.path.join(output_directory, subdir)))

    if LOO:
        epg_files, total_epg_secs = get_data_files_from_folder(os.path.join(animal_path,'EPG'), train_valid_split=False)
        valid_files, total_val_secs = get_data_files_from_folder(os.path.join(animal_path,'BL'), train_valid_split=False)
        # train_files = get_all_data_files(data_path, animal, train_valid_split=False)
        trained_files_filename = [i for i in os.listdir(run_logdir) if 'picked_train' in i][0]
        train_files = np.array(pd.read_csv(os.path.join(run_logdir, trained_files_filename)).values).reshape(-1)
    else:
        epg_files, total_epg_secs = get_data_files_from_folder(os.path.join(animal_path,'EPG'), train_valid_split=False)
        train_files, valid_files, total_train_secs, total_val_secs = get_data_files_from_folder(os.path.join(animal_path,'BL'), train_valid_split=True, train_percentage=0.8)

    # epg_set = csv_reader_dataset(epg_files, batch_size=batch_size, shuffle=False)
    epg_set = v2_create_dataset(epg_files, batch_size=batch_size, shuffle=False, repeat=False,
                          n_sec_per_sample=1, sr=512)
    valid_set = v2_create_dataset(valid_files, batch_size=batch_size, shuffle=False, repeat=False, n_sec_per_sample=1, sr=512)
    # valid_set = csv_reader_dataset(valid_files, batch_size=batch_size, shuffle=False)
    train_set = v2_create_dataset(train_files, batch_size=batch_size, shuffle=False, repeat=False, n_sec_per_sample=1, sr=512)
    # train_set = csv_reader_dataset(train_files, batch_size=batch_size, shuffle=False)
    total_epg_batches = np.int(len(epg_files) * 3600 / batch_size)  # given that n_sec_per_samp is 1
    total_valid_batches = np.int(len(valid_files) * 3600 / batch_size)
    total_train_batches = np.int(len(train_files) * 3600  / batch_size)

    encoder = tf.keras.models.load_model(os.path.join(run_logdir, 'encoder.h5'))
    decoder = tf.keras.models.load_model(os.path.join(run_logdir, 'decoder.h5'))
    # disc_x = tf.keras.models.load_model(run_logdir+'/discriminator_x.h5')


    def compute_batch_distance(z):
        distance = []
        for i in range(z.shape[0]):
            distance.append(scipy.spatial.distance.euclidean(z[i].numpy(),np.zeros(z_dim)))
        return np.array(distance)


    def  compute_distros(dataset, directory, total_batch, num2collect=10, name="epg"):
        """
        compute all related metrices and save some batches for future inspection
        :param dataset:
        :param directory:
        :param total_batch:
        :param num2collect:
        :return:
        """
        if not os.path.exists(directory):
            os.mkdir(directory)
        errors = np.array([])

        # probilities = np.array([])
        distances = np.array([])
        z_all = np.zeros(z_dim)
        filenames_all = []
        rat_ids_all = []

        coll_batch_inds = np.random.choice(total_batch, num2collect, replace=False)
        np.savetxt(os.path.join(directory, "randomly-selected-{}-{}batches-{}.csv".format(num2collect, total_batch, name)), np.array(np.sort(coll_batch_inds)).reshape(-1,1), fmt="%d", delimiter=",")
        print("{}/{} batches in {} to collect: {}".format(num2collect, total_batch, name, coll_batch_inds))
        for i, batch_data in enumerate(dataset):
            batch_features, batch_label, batch_fn, batch_rat_id = [batch_data[i]
                                                                   for i in
                                                                   range(
                                                                       len(
                                                                           batch_data))]
            z = encoder(batch_features)
            z_all = np.vstack((z_all, z.numpy()))
            batch_fn_array = np.array(batch_fn).astype(np.str)
            batch_label_array = np.array(batch_label).astype(np.str)
            batch_rat_id_array = np.array(batch_rat_id).astype(np.str)

            filenames_all.append(batch_fn_array)
            rat_ids_all.append(batch_rat_id_array)
    
            x_hat = decoder(z)
            # prob = scipy.special.expit(disc_x(x_hat)[0]).ravel()
            # probilities = np.concatenate((probilities,prob),axis=0)
    
            loss = np.square(batch_features - x_hat)[:, :, 0]
            # error = loss.reshape(loss.shape[0]*loss.shape[1])
            error = np.mean(loss, axis=1).ravel()
            errors = np.concatenate((errors, error), axis=0)
            
            # if i % 20 == 0:
            #     print("{}/{} batches in {}".format(i, total_batch, name))
    
            distance = compute_batch_distance(z)
            distances = np.concatenate((distances, distance), axis=0)
    
            if i in coll_batch_inds:
                batch_features_array = np.array(batch_features).astype(np.float32)
                x_hat_array = np.array(x_hat).astype(np.float32)
                coll_info = np.concatenate((
                    batch_fn_array.reshape(-1, 1),
                    batch_label_array.reshape(-1, 1),
                    batch_rat_id_array.reshape(-1, 1),
                    error.reshape(-1, 1),
                    distance.reshape(-1, 1),
                    batch_features_array.reshape(batch_size, -1),
                    x_hat_array.reshape(batch_size, -1)
                ), axis=1)
                np.savetxt(os.path.join(directory,
                                        "collected_info-[fn,lb,id,err,dist,eeg,recon]-{}.csv".format(
                                            i)), np.array(coll_info), fmt="%s",
                           delimiter=",")
                # print("Saved {}".format(os.path.join(directory,
                #                                         "collected_info-[fn,lb,id,err,dist,eeg,recon]-{}.csv".format(
                #                                             i))))
        gc.collect()
        
        np.save(os.path.join(directory, 'errors.npy'), errors)
        # np.save(directory+'/probilities.npy', probilities)
        np.save(os.path.join(directory, 'distances.npy'), distances)
        np.save(os.path.join(directory, 'z.npy'), z_all[1:,:])
        concat_info = np.concatenate((np.array(filenames_all).reshape(-1,1), np.array(rat_ids_all).reshape(-1,1)), axis=1)
        np.savetxt(os.path.join(directory, 'batch_samples_infomation.csv'), concat_info, delimiter=",", fmt="%s")
    
    if not precomputed:
        compute_distros(valid_set, os.path.join(output_directory, 'valid_data'), total_valid_batches, num2collect=total_valid_batches//10, name="valid")
        compute_distros(epg_set, os.path.join(output_directory, 'epg_data'), total_epg_batches, num2collect=total_epg_batches//10, name="epg")
        compute_distros(train_set, os.path.join(output_directory, 'train_data'), total_train_batches, num2collect=total_train_batches//10, name="train")

    ####################################################################

    t_errors = np.load(os.path.join(output_directory, 'train_data', 'errors.npy'))
    # t_probilities = np.load(output_directory+'train_data'+'/probilities.npy')
    t_distances = np.load(os.path.join(output_directory, 'train_data', 'distances.npy'))
    # t_z_all = np.load(output_directory+'train_data'+'/z.npy')

    v_errors = np.load(os.path.join(output_directory, 'valid_data', 'errors.npy'))
    # v_probilities = np.load(output_directory+'valid_data'+'/probilities.npy')
    v_distances = np.load(os.path.join(output_directory, 'valid_data', 'distances.npy'))
    # v_z_all = np.load(output_directory+'valid_data'+'/z.npy')

    e_errors = np.load(os.path.join(output_directory, 'epg_data', 'errors.npy'))
    # e_probilities = np.load(output_directory+'epg_data'+'/probilities.npy')
    e_distances = np.load(os.path.join(output_directory, 'epg_data', 'distances.npy'))
    # e_z_all = np.load(output_directory+'epg_data'+'/z.npy')

    fig = plt.figure(figsize=(10,10))
    sns.distplot(e_errors, kde=False, norm_hist=True, label='epg errors')
    sns.distplot(t_errors, kde=False, norm_hist=True, label='train errors')
    sns.distplot(v_errors, kde=False, norm_hist=True, label='valid errors')
    plt.yscale('log')
    plt.legend()
    plt.savefig(os.path.join(output_directory, 'errors-{}.png'.format(animal)))
    plt.savefig(os.path.join(output_directory, 'errors-{}.pdf'.format(animal)), format="pdf")
    plt.close()

    # fig = plt.figure(figsize=(10,10))
    # sns.distplot(e_probilities, kde=False, norm_hist=True, label='epg probabilities')
    # sns.distplot(t_probilities, kde=False, norm_hist=True, label='train probabilities')
    # sns.distplot(v_probilities, kde=False, norm_hist=True, label='valid probabilities')
    # plt.legend()
    # plt.savefig(output_directory+'probabilities.png')
    # plt.close()

    fig = plt.figure(figsize=(10,10))
    sns.distplot(e_distances, kde=False, norm_hist=True, label='epg distances')
    sns.distplot(t_distances, kde=False, norm_hist=True, label='train distances')
    sns.distplot(v_distances, kde=False, norm_hist=True, label='valid distances')
    plt.legend()
    plt.savefig(output_directory+'distances-{}.png'.format(animal))
    plt.savefig(output_directory+'distances-{}.pdf'.format(animal), format="pdf")
    plt.close()

    # reshaped_t_errors = np.reshape(t_errors, (int(t_errors.shape[0]/2560),2560))
    # whole_segment_t_errors = np.mean(reshaped_t_errors, axis = 1)
    whole_segment_t_errors = t_errors
    del t_errors
    np.save(os.path.join(output_directory, 'train_data', 'whole_segment_t_errors.npy'), whole_segment_t_errors)
    # whole_segment_t_errors = np.load(output_directory+'train_data'+'/whole_segment_t_errors.npy')
    moving_average = pd.Series(whole_segment_t_errors).rolling(3600*12).mean()
    moving_std = pd.Series(whole_segment_t_errors).rolling(3600*12).std()
    fig = plt.figure(figsize=(20,10))
    plt.scatter(np.arange(int(whole_segment_t_errors.shape[0])),whole_segment_t_errors, color='orange', label='errors', marker='.', s=1)  
    plt.plot(moving_average, linewidth=2, color='black', label='moving average')
    plt.fill_between(moving_std.index, (moving_average-moving_std), (moving_average+moving_std), color='red', alpha=.2, label=' moving std')
    # plt.ylim([0,2000])
    plt.xlabel('Time in days')
    plt.ylabel('Reconstruction error')
    ticks = np.arange(0,whole_segment_t_errors.shape[0], 3600*24)
    plt.xticks(ticks, np.arange(0,len(ticks), 1))
    plt.legend()
    plt.savefig(os.path.join(output_directory, 't_whole_segment_errors-{}.png'.format(animal)))
    plt.savefig(os.path.join(output_directory, 't_whole_segment_errors-{}.pdf'.format(animal)), format="pdf")
    plt.close()

    th90 = np.percentile(whole_segment_t_errors, 90)
    th95 = np.percentile(whole_segment_t_errors, 95)
    th99 = np.percentile(whole_segment_t_errors, 99)

    # reshaped_v_errors = np.reshape(v_errors, (int(v_errors.shape[0]/2560),2560))
    # whole_segment_v_errors = np.mean(reshaped_v_errors, axis = 1)
    whole_segment_v_errors = v_errors
    del v_errors
    np.save(os.path.join(output_directory, 'valid_data', 'whole_segment_v_errors.npy'), whole_segment_v_errors)
    # whole_segment_v_errors = np.load(output_directory+'valid_data'+'/whole_segment_v_errors.npy')
    moving_average = pd.Series(whole_segment_v_errors).rolling(3600*12).mean()
    moving_std = pd.Series(whole_segment_v_errors).rolling(3600*12).std()
    fig = plt.figure(figsize=(20,10))
    plt.scatter(np.arange(int(whole_segment_v_errors.shape[0])),whole_segment_v_errors, color='orange', label='errors', marker='.', s=1)  
    plt.plot(moving_average, linewidth=2, color='black', label='moving average')
    plt.fill_between(moving_std.index, (moving_average-moving_std), (moving_average+moving_std), color='red', alpha=.2, label=' moving std')
    plt.hlines([th90, th95, th99], 0, len(whole_segment_v_errors), colors='green', linewidth=2, linestyles='dashed', label='90th, 95th, 99th percentiles')
    # plt.ylim([0,2000])
    plt.xlabel('Time in days')
    plt.ylabel('Reconstruction error')
    ticks = np.arange(0,whole_segment_v_errors.shape[0], 3600*24)
    plt.xticks(ticks, np.arange(0,len(ticks), 1))
    plt.legend()
    plt.savefig(os.path.join(output_directory, 'v_whole_segment_errors-{}.png').format(animal))
    plt.savefig(os.path.join(output_directory, 'v_whole_segment_errors-{}.pdf'.format(animal)), format="pdf")
    plt.close()

    # reshaped_e_errors = np.reshape(e_errors, (int(e_errors.shape[0]/2560),2560))
    # whole_segment_e_errors = np.mean(reshaped_e_errors, axis = 1)
    whole_segment_e_errors = e_errors
    del e_errors
    np.save(os.path.join(output_directory, 'epg_data','whole_segment_e_errors.npy'), whole_segment_e_errors)
    # whole_segment_e_errors = np.load(output_directory+'epg_data'+'/whole_segment_e_errors.npy')
    moving_average = pd.Series(whole_segment_e_errors).rolling(3600*12).mean()
    moving_std = pd.Series(whole_segment_e_errors).rolling(3600*12).std()
    fig = plt.figure(figsize=(20,10))
    plt.scatter(np.arange(int(whole_segment_e_errors.shape[0])),whole_segment_e_errors, color='orange', label='errors', marker='.', s=1)
    plt.plot(moving_average, linewidth=2, color='black', label='moving average')
    plt.fill_between(moving_std.index, (moving_average-moving_std), (moving_average+moving_std), color='red', alpha=.2, label=' moving std')
    plt.hlines([th90, th95, th99], 0, len(whole_segment_e_errors), colors='green', linewidth=2, linestyles='dashed', label='90th, 95th, 99th percentiles')
    # plt.ylim([0,2000])
    plt.xlabel('Time in days')
    plt.ylabel('Reconstruction error')
    ticks = np.arange(0,whole_segment_e_errors.shape[0], 3600*24)
    plt.xticks(ticks, np.arange(0,len(ticks), 1)) 
    plt.legend()
    plt.savefig(os.path.join(output_directory, 'e_whole_segment_errors-{}.png'.format(animal)))
    plt.savefig(os.path.join(output_directory, 'e_whole_segment_errors-{}.pdf'.format(animal)), format="pdf")
    plt.close()

    fig = plt.figure(figsize=(10,10))
    sns.distplot(whole_segment_e_errors, kde=False, norm_hist=True, label='epg errors')
    sns.distplot(whole_segment_t_errors, kde=False, norm_hist=True, label='train errors')
    sns.distplot(whole_segment_v_errors, kde=False, norm_hist=True, label='valid errors')
    plt.legend()
    plt.savefig(os.path.join(output_directory,'whole_segment_errors-{}.png'.format(animal)))
    plt.savefig(os.path.join(output_directory,'whole_segment_errors-{}.pdf'.format(animal)), format="pdf")
    plt.close()

    th_whole_segment_t_errors = np.copy(whole_segment_t_errors)
    th_whole_segment_t_errors[th_whole_segment_t_errors < th99] = 0
    fig = plt.figure(figsize=(20,10))
    plt.scatter(np.arange(len(th_whole_segment_t_errors)),th_whole_segment_t_errors, marker='.', color='orange', s=1)
    plt.hlines(th99, 0, len(whole_segment_t_errors), colors='green', linewidth=1, linestyles='dashed', label='99th percentiles')
    # plt.ylim([0,2000])
    plt.xlabel('Time in days')
    plt.ylabel('Reconstruction error')
    ticks = np.arange(0,th_whole_segment_t_errors.shape[0], 3600*24)
    plt.xticks(ticks, np.arange(0,len(ticks), 1)) 
    plt.savefig(os.path.join(output_directory,'th_t_whole_segment_errors-{}.png'.format(animal)))
    plt.savefig(os.path.join(output_directory,'th_t_whole_segment_errors-{}.pdf'.format(animal)), format="pdf")
    plt.close()

    th_whole_segment_v_errors = np.copy(whole_segment_v_errors)
    th_whole_segment_v_errors[th_whole_segment_v_errors < th99] = 0
    fig = plt.figure(figsize=(20,10))
    plt.scatter(np.arange(len(th_whole_segment_v_errors)),th_whole_segment_v_errors, marker='.', color='orange', s=1)
    plt.hlines(th99, 0, len(whole_segment_v_errors), colors='green', linewidth=1, linestyles='dashed', label='99th percentiles')
    # plt.ylim([0,2000])
    plt.xlabel('Time in days')
    plt.ylabel('Reconstruction error')
    ticks = np.arange(0,whole_segment_v_errors.shape[0], 3600*24)
    plt.xticks(ticks, np.arange(0,len(ticks), 1)) 
    plt.savefig(os.path.join(output_directory,'th_v_whole_segment_errors-{}.png'.format(animal)))
    plt.savefig(os.path.join(output_directory,'th_v_whole_segment_errors-{}.pdf'.format(animal)), format="pdf")
    plt.close()

    th_whole_segment_e_errors = np.copy(whole_segment_e_errors)
    th_whole_segment_e_errors[th_whole_segment_e_errors < th99] = 0
    fig = plt.figure(figsize=(20,10))
    plt.scatter(np.arange(len(th_whole_segment_e_errors)),th_whole_segment_e_errors, marker='.', color='orange', s=1)
    plt.hlines(th99, 0, len(whole_segment_e_errors), colors='green', linewidth=1, linestyles='dashed', label='99th percentiles')
    # plt.ylim([0,2000])
    plt.xlabel('Time in days')
    plt.ylabel('Reconstruction error')
    ticks = np.arange(0,whole_segment_e_errors.shape[0], 3600*24)
    plt.xticks(ticks, np.arange(0,len(ticks), 1)) 
    plt.savefig(os.path.join(output_directory,'th_e_whole_segment_errors-{}.png'.format(animal)))
    plt.savefig(os.path.join(output_directory,'th_e_whole_segment_errors-{}.pdf'.format(animal)), format="pdf")
    plt.close()

    th90_d = np.percentile(t_distances, 90)
    th95_d = np.percentile(t_distances, 95)
    th99_d = np.percentile(t_distances, 99)

    moving_average = pd.Series(t_distances).rolling(3600*12).mean()
    moving_std = pd.Series(t_distances).rolling(3600*12).std()
    fig = plt.figure(figsize=(20,10))
    plt.scatter(np.arange(int(t_distances.shape[0])),t_distances, color='orange', label='errors', marker='.', s=1)  
    plt.plot(moving_average, linewidth=2, color='black', label='moving average')
    plt.fill_between(moving_std.index, (moving_average-moving_std), (moving_average+moving_std), color='red', alpha=.2, label=' moving std')
    # plt.ylim([0,4])
    plt.xlabel('Time in days')
    plt.ylabel('Reconstruction error')
    ticks = np.arange(0,t_distances.shape[0], 3600*24)
    plt.xticks(ticks, np.arange(0,len(ticks), 1))
    plt.legend()
    plt.savefig(os.path.join(output_directory,'t_distances-{}.png'.format(animal)))
    plt.savefig(os.path.join(output_directory,'t_distances-{}.pdf'.format(animal)), format="pdf")
    plt.close()

    moving_average = pd.Series(v_distances).rolling(3600*12).mean()
    moving_std = pd.Series(v_distances).rolling(3600*12).std()
    fig = plt.figure(figsize=(20,10))
    plt.scatter(np.arange(int(v_distances.shape[0])),v_distances, color='orange', label='errors', marker='.', s=1)  
    plt.plot(moving_average, linewidth=2, color='black', label='moving average')
    plt.fill_between(moving_std.index, (moving_average-moving_std), (moving_average+moving_std), color='red', alpha=.2, label=' moving std')
    plt.hlines([th90_d, th95_d, th99_d], 0, len(v_distances), colors='green', linewidth=2, linestyles='dashed', label='90th, 95th, 99th percentiles')
    # plt.ylim([0,4])
    plt.xlabel('Time in hours')
    plt.ylabel('Reconstruction error')
    ticks = np.arange(0,v_distances.shape[0], 3600*24)
    plt.xticks(ticks, np.arange(0,len(ticks), 1))
    plt.legend()
    plt.savefig(os.path.join(output_directory,'v_distances-{}.png'.format(animal)))
    plt.savefig(os.path.join(output_directory,'v_distances-{}.pdf'.format(animal)), format="pdf")
    plt.close()

    moving_average = pd.Series(e_distances).rolling(3600*12).mean()
    moving_std = pd.Series(e_distances).rolling(3600*12).std()
    fig = plt.figure(figsize=(20,10))
    plt.scatter(np.arange(int(e_distances.shape[0])),e_distances, color='orange', label='errors', marker='.', s=1)
    plt.plot(moving_average, linewidth=2, color='black', label='moving average')
    plt.fill_between(moving_std.index, (moving_average-moving_std), (moving_average+moving_std), color='red', alpha=.2, label=' moving std')
    plt.hlines([th90_d, th95_d, th99_d], 0, len(e_distances), colors='green', linewidth=2, linestyles='dashed', label='90th, 95th, 99th percentiles')
    # plt.ylim([0,4])
    plt.xlabel('Time in days')
    plt.ylabel('Reconstruction error')
    ticks = np.arange(0,e_distances.shape[0], 3600*24)
    plt.xticks(ticks, np.arange(0,len(ticks), 1)) 
    plt.legend()
    plt.savefig(os.path.join(output_directory,'e_distances-{}.png'.format(animal)))
    plt.savefig(os.path.join(output_directory,'e_distances-{}.pdf'.format(animal)), format="pdf")
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
        ticks = np.arange(0,frequency_t.shape[0], (3600)*24)
        plt.xticks(ticks, np.arange(0,len(ticks), 1)) 
        plt.savefig(os.path.join(output_directory,'frequency_t_'+str(window_in_minutes)+' minutes_no-{}.png'.format(animal)))
        plt.savefig(os.path.join(output_directory,'frequency_t_'+str(window_in_minutes)+' minutes_no-{}.pdf'.format(animal)), format="pdf")
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
        ticks = np.arange(0,frequency_v.shape[0], (3600)*24)
        plt.xticks(ticks, np.arange(0,len(ticks), 1)) 
        plt.savefig(os.path.join(output_directory,'frequency_v_'+str(window_in_minutes)+' minutes_no-{}.png'.format(animal)))
        plt.savefig(os.path.join(output_directory,'frequency_v_'+str(window_in_minutes)+' minutes_no-{}.pdf'.format(animal)), format="pdf")
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
        ticks = np.arange(0,frequency_e.shape[0], (3600)*24)
        plt.xticks(ticks, np.arange(0,len(ticks), 1)) 
        plt.savefig(os.path.join(output_directory,'frequency_e_'+str(window_in_minutes)+' minutes_no-{}.png'.format(animal)))
        plt.savefig(os.path.join(output_directory,'frequency_e_'+str(window_in_minutes)+' minutes_no-{}.pdf'.format(animal)), format="pdf")
        plt.close()

    #############################################################################
