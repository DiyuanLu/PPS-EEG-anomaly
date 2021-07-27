import os
import scipy

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
import pandas as pd
from sklearn.decomposition import PCA
from input_pipeline import csv_reader_dataset, get_all_data_files, get_data_files_from_folder, v2_create_dataset
from utils import get_timestamp_from_file


def scan_animals_with_pretrained_model(args):
    """
    Scan the whole data with a pretrained model
    :param args:
    :return:
    """
    batch_size = 800
    z_dim = args.z_dim

    # for model_name in args.models[:]:
    print('working on: '+ args.model_dir)
    animal = os.path.basename(args.model_dir).split("_")[-1]
    # animal = model_name[24:]
    if "326" in animal:
        data_path = args.ctrl_data_path
    else:
        data_path = args.pps_data_path
    #
    animal_path = os.path.join(data_path, animal, animal)

    # run_logdir = os.path.join(args.root_logdir, os.path.basename(args.model_dir))
    output_directory = args.run_logdir +  '/stats_{}/'.format(animal)


    if not os.path.exists(output_directory):  # if subfolder doesn't exist, should make the directory and then save file.
        os.makedirs(output_directory)

    if args.LOO:
        epg_files = get_data_files_from_folder(animal_path + '/EPG/',
                                               train_valid_split=False)
        train_val_files = get_data_files_from_folder(animal_path + '/BL/',
                                                 train_valid_split=False)

        # trained_files_filename = \
        # [i for i in os.listdir(args.model_dir) if 'picked_train' in i][0]
        
    else:
        epg_files = get_data_files_from_folder(animal_path + '/EPG/',
                                               train_valid_split=False)
        train_files, valid_files = get_data_files_from_folder(
            animal_path + '/BL/')

    # epg_set = csv_reader_dataset(epg_files, batch_size=batch_size,
    #                              shuffle=False)
    # valid_set = csv_reader_dataset(valid_files, batch_size=batch_size,
    #                                shuffle=False)
    # train_set = csv_reader_dataset(train_files, batch_size=batch_size,
    #                                shuffle=False)
    np.savetxt(os.path.join(args.run_logdir,
                            "{}_picked_EPG_files_{}.csv".format(args.LOO_animal, len(epg_files))),
               np.array(epg_files), fmt="%s",
               delimiter=",")
    np.savetxt(os.path.join(args.run_logdir,
                            "{}_picked_BL_files_{}.csv".format(args.LOO_animal, len(train_val_files))),
               np.array(train_val_files), fmt="%s",
               delimiter=",")
    epg_set = v2_create_dataset(epg_files, batch_size=batch_size,
                                 shuffle=False, n_sec_per_sample=args.n_sec_per_sample, sr=args.sampling_rate)
    # valid_set = v2_create_dataset(valid_files, batch_size=batch_size,
    #                                shuffle=False, n_sec_per_sample=args.n_sec_per_sample, sr=512)
    train_set = v2_create_dataset(train_val_files, batch_size=batch_size, shuffle=False,
                          n_sec_per_sample=args.n_sec_per_sample, sr=args.sampling_rate)

    total_num_batches_epg = (len(epg_files) * 720 * 5) // (args.n_sec_per_sample * batch_size)
    # total_num_batches_valid = (len(valid_files) * 720 * 5) // (args.n_sec_per_sample * batch_size)
    total_num_batches_train = (len(train_val_files) * 720 * 5) // (args.n_sec_per_sample * batch_size)
    hour_span_epg = (get_timestamp_from_file(os.path.basename(epg_files[-1])) - get_timestamp_from_file(os.path.basename(epg_files[0]))) / 3600
    # hour_span_valid = (get_timestamp_from_file(valid_files[-1]) - get_timestamp_from_file(valid_files[0])) / 3600
    hour_span_train = (get_timestamp_from_file(os.path.basename(train_val_files[-1])) - get_timestamp_from_file(os.path.basename(train_val_files[0]))) / 3600

    encoder = tf.keras.models.load_model(os.path.join(args.model_dir, 'encoder.h5'), compile=True)
    decoder = tf.keras.models.load_model(os.path.join(args.model_dir, 'decoder.h5'), compile=True)

    # disc_x = tf.keras.models.load_model(run_logdir+'/discriminator_x.h5')
    def compute_batch_distance(z):
        distance = []
        for i in range(z.shape[0]):
            distance.append(scipy.spatial.distance.euclidean(z[i].numpy(),
                                                             np.zeros(z_dim)))
        return np.array(distance)

    def compute_distros(dataset, directory, total_batch=10, num2coll=0, name="train_data"):
        """
        compute the reconstruction errors, distances, and the latent code
        :param dataset:
        :param directory:
        :param total_batch:
        :param num2coll: how many random batches to collect for further visualization
        :return:
        """

        if not os.path.exists(directory):
            os.mkdir(directory)
        errors = np.array([])
        # probilities = np.array([])
        distances = np.array([])
        all_filenames = np.empty((0, 1))
        rat_ids = np.array([])
        labels = np.array([])
        z_all = np.zeros(z_dim)

        coll_batch_inds = np.random.choice(total_batch, num2coll, replace=False)
        for i, batch_data in enumerate(dataset):
            batch_features, batch_label, batch_fn, batch_rat_id = [batch_data[i]
                                                                   for i in
                                                                   range(
                                                                       len(
                                                                           batch_data))]
            z = encoder(batch_features)
            z_all = np.vstack((z_all, z.numpy()))
    
            x_hat = decoder(z)
            # prob = scipy.special.expit(disc_x(x_hat)[0]).ravel()
            # probilities = np.concatenate((probilities,prob),axis=0)
    
            loss = np.square(batch_features - x_hat)[:, :, 0]
            # error = loss.reshape(loss.shape[0]*loss.shape[1])
            error = np.mean(loss, axis=1).ravel()
            errors = np.concatenate((errors, error), axis=0)
    
            distance = compute_batch_distance(z)
            distances = np.concatenate((distances, distance), axis=0)

            all_filenames = np.vstack((all_filenames, batch_fn.numpy().reshape(-1,1)))
    
            if (i + 1) % (total_batch//10) == 0:
                print('finished: {} out of {} batches'.format(i, total_batch))
                
            if i in coll_batch_inds:
                coll_info = np.concatenate((
                    batch_fn.numpy().reshape(-1,1),
                    batch_label.numpy().reshape(-1,1),
                    batch_rat_id.numpy().astype(str).reshape(-1,1),
                    error.reshape(-1,1),
                    distance.reshape(-1,1),
                    batch_features.numpy().reshape(batch_size,-1),
                    x_hat.numpy().reshape(batch_size,-1)
                    ), axis=1)
                np.savetxt(os.path.join(directory, "{}-collected_info-[fn,lb,id,err,dist,eeg,recon]-{}.csv".format(args.LOO_animal,i)), np.array(coll_info), fmt="%s", delimiter=",")
                coll_info = 0
                
        np.save(directory + '/errors.npy', errors)
        # np.save(directory+'/probilities.npy', probilities)
        np.save(directory + '/distances.npy', distances)
        np.save(directory + '/z.npy', z_all[1:, :])
        np.savetxt(os.path.join(directory,
                                "all_filenames.csv"), np.array(all_filenames),
                   fmt="%s", delimiter=",")

    compute_distros(epg_set, output_directory + 'epg_data', total_batch=total_num_batches_epg, num2coll=total_num_batches_epg//10)
    # compute_distros(valid_set, output_directory + 'valid_data', total_batch=total_num_batches_valid, num2coll=hour_span_valid//10)
    compute_distros(train_set, output_directory + 'train_data', total_batch=total_num_batches_train, num2coll=total_num_batches_train//10)

    ####################################################################
    t_errors = np.load(output_directory + 'train_data' + '/errors.npy')
    # t_probilities = np.load(output_directory+'train_data'+'/probilities.npy')
    t_distances = np.load(output_directory + 'train_data' + '/distances.npy')
    # t_z_all = np.load(output_directory+'train_data'+'/z.npy')

    # v_errors = np.load(output_directory + 'valid_data' + '/errors.npy')
    # # v_probilities = np.load(output_directory+'valid_data'+'/probilities.npy')
    # v_distances = np.load(output_directory + 'valid_data' + '/distances.npy')
    # v_z_all = np.load(output_directory+'valid_data'+'/z.npy')

    e_errors = np.load(output_directory + 'epg_data' + '/errors.npy')
    # e_probilities = np.load(output_directory+'epg_data'+'/probilities.npy')
    e_distances = np.load(output_directory + 'epg_data' + '/distances.npy')
    # e_z_all = np.load(output_directory+'epg_data'+'/z.npy')

    fig = plt.figure(figsize=(10, 10))
    sns.distplot(e_errors, kde=False, norm_hist=True, label='epg errors')
    sns.distplot(t_errors, kde=False, norm_hist=True, label='train errors')
    # sns.distplot(v_errors, kde=False, norm_hist=True, label='valid errors')
    plt.yscale('log')
    plt.legend()
    plt.savefig(output_directory + 'errors.png')
    plt.savefig(output_directory + 'errors.pdf', format="pdf")
    plt.close()

    # fig = plt.figure(figsize=(10,10))
    # sns.distplot(e_probilities, kde=False, norm_hist=True, label='epg probabilities')
    # sns.distplot(t_probilities, kde=False, norm_hist=True, label='train probabilities')
    # sns.distplot(v_probilities, kde=False, norm_hist=True, label='valid probabilities')
    # plt.legend()
    # plt.savefig(output_directory+'probabilities.png')
    # plt.close()

    fig = plt.figure(figsize=(10, 10))
    sns.distplot(e_distances, kde=False, norm_hist=True,
                 label='epg distances')
    sns.distplot(t_distances, kde=False, norm_hist=True,
                 label='train distances')
    # sns.distplot(v_distances, kde=False, norm_hist=True,
    #              label='valid distances')
    plt.legend()
    plt.savefig(output_directory + 'distances.png')
    plt.savefig(output_directory + 'distances.pdf')
    plt.close()

    # reshaped_t_errors = np.reshape(t_errors, (int(t_errors.shape[0]/2560),2560))
    # whole_segment_t_errors = np.mean(reshaped_t_errors, axis = 1)
    # whole_segment_t_errors = t_errors
    # del t_errors
    # np.save(output_directory + 'train_data' + '/whole_segment_t_errors.npy',
    #         whole_segment_t_errors)
    # # whole_segment_t_errors = np.load(output_directory+'train_data'+'/whole_segment_t_errors.npy')
    # moving_average = pd.Series(whole_segment_t_errors).rolling(
    #     3600 * 12).mean()
    # moving_std = pd.Series(whole_segment_t_errors).rolling(3600 * 12).std()
    # fig = plt.figure(figsize=(20, 10))
    # plt.scatter(np.arange(int(whole_segment_t_errors.shape[0])),
    #             whole_segment_t_errors, color='orange', label='errors',
    #             marker='.', s=1)
    # plt.plot(moving_average, linewidth=2, color='black',
    #          label='moving average')
    # plt.fill_between(moving_std.index, (moving_average - moving_std),
    #                  (moving_average + moving_std), color='red', alpha=.2,
    #                  label=' moving std')
    # # plt.ylim([0,2000])
    # plt.xlabel('Time in days')
    # plt.ylabel('Reconstruction error')
    # ticks = np.arange(0, whole_segment_t_errors.shape[0], 3600 * 24)
    # plt.xticks(ticks, np.arange(0, len(ticks), 1))
    # plt.legend()
    # plt.savefig(output_directory + 't_whole_segment_errors.png')
    # plt.savefig(output_directory + 't_whole_segment_errors.pdf')
    # plt.close()

    # reshaped_v_errors = np.reshape(v_errors, (int(v_errors.shape[0]/2560),2560))
    # inspect_data = np.mean(v_errors, axis=1)

    # reshaped_e_errors = np.reshape(e_errors, (int(e_errors.shape[0]/2560),2560))
    # whole_segment_e_errors = np.mean(reshaped_e_errors, axis = 1)
    
    
    # whole_segment_v_errors = quantile_analysis_on_errors(output_directory, hlines_loc=[th90, th95,th99], name="valid")
    whole_segment_t_errors = quantile_analysis_on_errors(t_errors, output_directory,
                                                         hlines_loc=None,
                                                         name="train")
    th90 = np.percentile(whole_segment_t_errors, 90)
    th95 = np.percentile(whole_segment_t_errors, 95)
    th99_e = np.percentile(whole_segment_t_errors, 99)
    whole_segment_e_errors = quantile_analysis_on_errors(e_errors, output_directory,
                                                         hlines_loc=[th90, th95,
                                                                     th99_e],
                                                         name="epg")
    
    # whole_segment_e_errors = e_errors
    # del e_errors
    # np.save(output_directory + 'epg_data' + '/whole_segment_e_errors.npy',
    #         whole_segment_e_errors)
    # # whole_segment_e_errors = np.load(output_directory+'epg_data'+'/whole_segment_e_errors.npy')
    # moving_average = pd.Series(whole_segment_e_errors).rolling(
    #     3600 * 12).mean()
    # moving_std = pd.Series(whole_segment_e_errors).rolling(3600 * 12).std()
    # fig = plt.figure(figsize=(20, 10))
    # plt.scatter(np.arange(int(whole_segment_e_errors.shape[0])),
    #             whole_segment_e_errors, color='orange', label='errors',
    #             marker='.', s=1)
    # plt.plot(moving_average, linewidth=2, color='black',
    #          label='moving average')
    # plt.fill_between(moving_std.index, (moving_average - moving_std),
    #                  (moving_average + moving_std), color='red', alpha=.2,
    #                  label=' moving std')
    # plt.hlines([th90, th95, th99], 0, len(whole_segment_e_errors),
    #            colors='green', linewidth=2, linestyles='dashed',
    #            label='90th, 95th, 99th percentiles')
    # # plt.ylim([0,2000])
    # plt.xlabel('Time in days')
    # plt.ylabel('Reconstruction error')
    # ticks = np.arange(0, whole_segment_e_errors.shape[0], 3600 * 24)
    # plt.xticks(ticks, np.arange(0, len(ticks), 1))
    # plt.legend()
    # plt.savefig(output_directory + 'e_whole_segment_errors.png')
    # plt.savefig(output_directory + 'e_whole_segment_errors.pdf')
    # plt.close()

    fig = plt.figure(figsize=(10, 10))
    sns.distplot(whole_segment_e_errors, kde=False, norm_hist=True,
                 label='epg errors')
    sns.distplot(whole_segment_t_errors, kde=False, norm_hist=True,
                 label='train errors')
    # sns.distplot(whole_segment_v_errors, kde=False, norm_hist=True,
    #              label='valid errors')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_directory + 'whole_segment_errors.png')
    plt.savefig(output_directory + 'whole_segment_errors.pdf', format="pdf")
    plt.close()
    
    ## plot percentile plots of reconstructions errors
    plot_scatter_99th_quantile_error_segs(whole_segment_t_errors, th99_e, output_directory, name="train")
    # plot_scatter_99th_quantile_error_segs(whole_segment_v_errors, th99, output_directory, name="valid")
    plot_scatter_99th_quantile_error_segs(whole_segment_e_errors, th99_e, output_directory, name="epg")

    # th_whole_segment_v_errors = np.copy(whole_segment_v_errors)
    # th_whole_segment_v_errors[th_whole_segment_v_errors < th99] = 0
    # fig = plt.figure(figsize=(20, 10))
    # plt.scatter(np.arange(len(th_whole_segment_v_errors)),
    #             th_whole_segment_v_errors, marker='.', color='orange', s=1)
    # plt.hlines(th99, 0, len(whole_segment_v_errors), colors='green',
    #            linewidth=1, linestyles='dashed', label='99th percentiles')
    # # plt.ylim([0,2000])
    # plt.xlabel('Time in days')
    # plt.ylabel('Reconstruction error')
    # ticks = np.arange(0, whole_segment_v_errors.shape[0], 3600 * 24)
    # plt.xticks(ticks, np.arange(0, len(ticks), 1))
    # plt.tight_layout()
    # plt.savefig(output_directory + 'th_v_whole_segment_errors.png')
    # plt.savefig(output_directory + 'th_v_whole_segment_errors.pdf')
    # plt.close()

    # th_whole_segment_e_errors = np.copy(whole_segment_e_errors)
    # th_whole_segment_e_errors[th_whole_segment_e_errors < th99] = 0
    # fig = plt.figure(figsize=(20, 10))
    # plt.scatter(np.arange(len(th_whole_segment_e_errors)),
    #             th_whole_segment_e_errors, marker='.', color='orange', s=1)
    # plt.hlines(th99, 0, len(whole_segment_e_errors), colors='green',
    #            linewidth=1, linestyles='dashed', label='99th percentiles')
    # # plt.ylim([0,2000])
    # plt.xlabel('Time in days')
    # plt.ylabel('Reconstruction error')
    # ticks = np.arange(0, whole_segment_e_errors.shape[0], 3600 * 24)
    # plt.xticks(ticks, np.arange(0, len(ticks), 1))
    # plt.tight_layout()
    # plt.savefig(output_directory + 'th_e_whole_segment_errors.png')
    # plt.savefig(output_directory + 'th_e_whole_segment_errors.pdf')
    # plt.close()

    th90_d = np.percentile(t_distances, 90)
    th95_d = np.percentile(t_distances, 95)
    th99_d = np.percentile(t_distances, 99)

    moving_average_error_analysis(t_distances, output_directory, hlines_loc=None, name="train")
    moving_average_error_analysis(t_distances, output_directory, hlines_loc=[th90_d, th95_d, th99_d], name="epg")

    # moving_average = pd.Series(v_distances).rolling(3600 * 12).mean()
    # moving_std = pd.Series(v_distances).rolling(3600 * 12).std()
    # fig = plt.figure(figsize=(20, 10))
    # plt.scatter(np.arange(int(v_distances.shape[0])), v_distances,
    #             color='orange', label='errors', marker='.', s=1)
    # plt.plot(moving_average, linewidth=2, color='black',
    #          label='moving average')
    # plt.fill_between(moving_std.index, (moving_average - moving_std),
    #                  (moving_average + moving_std), color='red', alpha=.2,
    #                  label=' moving std')
    # plt.hlines([th90_d, th95_d, th99_d], 0, len(v_distances), colors='green',
    #            linewidth=2, linestyles='dashed',
    #            label='90th, 95th, 99th percentiles')
    # # plt.ylim([0,4])
    # plt.xlabel('Time in hours')
    # plt.ylabel('Reconstruction error')
    # ticks = np.arange(0, v_distances.shape[0], 3600 * 24)
    # plt.xticks(ticks, np.arange(0, len(ticks), 1))
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(output_directory + 'v_distances.png')
    # plt.savefig(output_directory + 'v_distances.pdf')
    # plt.close()

    # moving_average = pd.Series(e_distances).rolling(3600 * 12).mean()
    # moving_std = pd.Series(e_distances).rolling(3600 * 12).std()
    # fig = plt.figure(figsize=(20, 10))
    # plt.scatter(np.arange(int(e_distances.shape[0])), e_distances,
    #             color='orange', label='errors', marker='.', s=1)
    # plt.plot(moving_average, linewidth=2, color='black',
    #          label='moving average')
    # plt.fill_between(moving_std.index, (moving_average - moving_std),
    #                  (moving_average + moving_std), color='red', alpha=.2,
    #                  label=' moving std')
    # plt.hlines([th90_d, th95_d, th99_d], 0, len(e_distances), colors='green',
    #            linewidth=2, linestyles='dashed',
    #            label='90th, 95th, 99th percentiles')
    # # plt.ylim([0,4])
    # plt.xlabel('Time in days')
    # plt.ylabel('Reconstruction error')
    # ticks = np.arange(0, e_distances.shape[0], 3600 * 24)
    # plt.xticks(ticks, np.arange(0, len(ticks), 1))
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig(output_directory + 'e_distances.png')
    # plt.savefig(output_directory + 'e_distances.pdf')
    # plt.close()

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


    test_window_in_minutes = [1, 5, 10, 15, 30, 60]
    for window_in_minutes in test_window_in_minutes:
        window = int((window_in_minutes * 60) / 5)
        ## r = np.reshape(th_whole_segment_t_errors[:len(th_whole_segment_t_errors)//window*window], (-1, window))
        ## frequency_t = np.sum(np.where(r>0, 1, 0),axis=1)
        r = np.where(whole_segment_t_errors > th99_e, 1, 0)
        frequency_t = pd.Series(r).rolling(window).sum()
        # fig = plt.figure(figsize=(20, 10))
        # plt.plot(np.arange(len(frequency_t)), frequency_t)
        # plt.xlabel('Time in days')
        # plt.ylabel('#suprathrehold segments per ' + str(
        #     window_in_minutes) + ' minutes')
        # ticks = np.arange(0, frequency_t.shape[0], (3600) * 24)
        # plt.xticks(ticks, np.arange(0, len(ticks), 1))
        # plt.tight_layout()
        # plt.savefig(output_directory + 'frequency_t_' + str(
        #     window_in_minutes) + ' minutes_no.png')
        # plt.savefig(output_directory + 'frequency_t_' + str(
        #     window_in_minutes) + ' minutes_no.pdf')
        #
        # plt.close()

        th99_frequency = np.nanpercentile(frequency_t, 99)

        ## r = np.reshape(th_whole_segment_v_errors[:len(th_whole_segment_v_errors)//window*window], (-1, window))
        ## frequency_v = np.sum(np.where(r>0, 1, 0),axis=1)
        suprathreshold_freq_analysis(whole_segment_t_errors, output_directory, th99_e, th99_frequency,
                                     window, window_in_minutes, name="train")
        # suprathreshold_freq_analysis(th_whole_segment_v_errors, output_directory, th99_e, th99_frequency,
        #                              window, window_in_minutes, name="valid")
        suprathreshold_freq_analysis(whole_segment_e_errors, output_directory, th99_e, th99_frequency,
                                     window, window_in_minutes, name="epg")

        ## r = np.reshape(th_whole_segment_e_errors[:len(th_whole_segment_e_errors)//window*window], (-1, window))
        ## frequency_e = np.sum(np.where(r>0, 1, 0),axis=1)
        # r = np.where(whole_segment_e_errors > th99_e, 1, 0)
        # frequency_e = pd.Series(r).rolling(window).sum()
        # fig = plt.figure(figsize=(20, 10))
        # plt.plot(np.arange(len(frequency_e)), frequency_e)
        # plt.axhline(th99_frequency, c='g', linewidth=3, linestyle='dotted')
        # if np.any(frequency_e.values > th99_frequency):
        #     plt.axvline(np.where(frequency_e > th99_frequency)[0][0], c='r',
        #                 linewidth=3, linestyle='dashed')
        # plt.xlabel('Time in days')
        # plt.ylabel('#suprathrehold segments per' + str(
        #     window_in_minutes) + ' minutes')
        # ticks = np.arange(0, frequency_e.shape[0], (3600) * 24)
        # plt.xticks(ticks, np.arange(0, len(ticks), 1))
        # plt.tight_layout()
        # plt.savefig(output_directory + 'frequency_e_' + str(
        #     window_in_minutes) + ' minutes_no.png')
        # plt.savefig(output_directory + 'frequency_e_' + str(
        #     window_in_minutes) + ' minutes_no.pdf')
        # plt.close()

    #############################################################################


def suprathreshold_freq_analysis(errors, output_directory, th99, th99_frequency, window,
                                 window_in_minutes, name="train"):
    r = np.where(errors > th99, 1, 0)
    frequency = pd.Series(r).rolling(window).sum()
    fig = plt.figure(figsize=(20, 10))
    plt.plot(np.arange(len(frequency)), frequency)

    if name != "train":
        plt.axhline(th99_frequency, c='g', linewidth=3, linestyle='dotted')
        if np.any(frequency.values > th99_frequency):
            plt.axvline(np.where(frequency > th99_frequency)[0][0], c='r',
                        linewidth=3, linestyle='dashed')
    plt.xlabel('Time in days')
    plt.ylabel('#suprathrehold segments per ' + str(
        window_in_minutes) + ' minutes')
    ticks = np.arange(0, frequency.shape[0], (3600) * 24)
    plt.xticks(ticks, np.arange(0, len(ticks), 1))
    plt.tight_layout()
    plt.savefig(output_directory + 'frequency_{}_{}_minutes_no.png'.format(name, window_in_minutes))
    plt.savefig(output_directory + 'frequency_{}_{}_minutes_no.pdf'.format(name, window_in_minutes), format="pdf")
    plt.close()


def moving_average_error_analysis(distances, output_directory, hlines_loc=None, name="train"):
    moving_average = pd.Series(distances).rolling(3600 * 12).mean()
    moving_std = pd.Series(distances).rolling(3600 * 12).std()
    fig = plt.figure(figsize=(20, 10))
    plt.scatter(np.arange(int(distances.shape[0])), distances,
                color='orange', label='errors', marker='.', s=1)
    plt.plot(moving_average, linewidth=2, color='black',
             label='moving average')
    plt.fill_between(moving_std.index, (moving_average - moving_std),
                     (moving_average + moving_std), color='red', alpha=.2,
                     label=' moving std')
    # plt.ylim([0,4])
    if hlines_loc:
        plt.hlines(hlines_loc, 0, len(distances), colors='green',
                              linewidth=2, linestyles='dashed',
                              label='90th, 95th, 99th percentiles')
    plt.xlabel('Time in days')
    plt.ylabel('Distance to the prior')
    ticks = np.arange(0, distances.shape[0], 3600 * 24)
    plt.xticks(ticks, np.arange(0, len(ticks), 1))
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_directory + '{}_distances.png'.format(name))
    plt.savefig(output_directory + '{}_distances.pdf'.format(name), format="pdf")
    plt.close()


def plot_scatter_99th_quantile_error_segs(seg_erros, th99, output_directory, name="train"):
    seg_erros[seg_erros < th99] = 0
    fig = plt.figure(figsize=(20, 10))
    plt.scatter(np.arange(len(seg_erros)),
                seg_erros, marker='.', color='orange', s=1)
    plt.hlines(th99, 0, len(seg_erros), colors='green',
               linewidth=1, linestyles='dashed', label='99th percentiles')
    # plt.ylim([0,2000])
    plt.xlabel('Time in days')
    plt.ylabel('Reconstruction error')
    ticks = np.arange(0, seg_erros.shape[0], 3600 * 24)
    plt.xticks(ticks, np.arange(0, len(ticks), 1))
    plt.tight_layout()
    plt.savefig(output_directory + '99th_{}_whole_segment_errors.png'.format(name))
    plt.savefig(output_directory + '99th_{}_whole_segment_errors.pdf'.format(name), format="pdf")
    plt.close()


def quantile_analysis_on_errors(inspect_data, output_directory, hlines_loc=None, name="valid"):
    
    # TODO: commented out
    # inspect_data = v_errors
    # del v_errors
    np.save(output_directory + '{}_data'.format(name) + '/whole_segment_{}_errors.npy'.format(name[0]),
            inspect_data)
    # whole_segment_v_errors = np.load(output_directory+'valid_data'+'/whole_segment_v_errors.npy')
    moving_average = pd.Series(inspect_data).rolling(
        3600 * 12).mean()
    moving_std = pd.Series(inspect_data).rolling(3600 * 12).std()
    fig = plt.figure(figsize=(20, 10))
    plt.scatter(np.arange(int(inspect_data.shape[0])),
                inspect_data, color='orange', label='errors',
                marker='.', s=1)
    plt.plot(moving_average, linewidth=2, color='black',
             label='moving average')
    plt.fill_between(moving_std.index, (moving_average - moving_std),
                     (moving_average + moving_std), color='red', alpha=.2,
                     label=' moving std')
    if hlines_loc:
        plt.hlines(hlines_loc, 0, len(inspect_data),
                   colors='green', linewidth=2, linestyles='dashed',
                   label='90th, 95th, 99th percentiles')
    # plt.ylim([0,2000])
    plt.xlabel('Time in days')
    plt.ylabel('Reconstruction error')
    ticks = np.arange(0, inspect_data.shape[0], 3600 * 24)
    plt.xticks(ticks, np.arange(0, len(ticks), 1))
    plt.legend()
    plt.savefig(output_directory + '{}_whole_segment_errors.png'.format(name))
    plt.savefig(output_directory + '{}_whole_segment_errors.pdf'.format(name), format="pdf")
    plt.close()
    return inspect_data

