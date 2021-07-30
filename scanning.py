import os
import scipy

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
import pandas as pd
from sklearn.decomposition import PCA
from input_pipeline import csv_reader_dataset, get_all_data_files, get_data_files_from_folder, v2_create_dataset, csv_reader_dataset3
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
        valid_files = get_data_files_from_folder(animal_path + '/BL/',
                                                 train_valid_split=False)

        trained_files_filename = \
        [i for i in os.listdir(args.model_dir) if 'picked_train' in i]
        trained_files = []
        for fn in trained_files_filename:
            fns = pd.read_csv(os.path.join(args.model_dir, fn), header=None).values
            trained_files += list(fns)
        trained_files = list(np.array(trained_files).reshape(-1))
        
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
                            "{}_picked_BL_files_{}.csv".format(args.LOO_animal, len(valid_files))),
               np.array(valid_files), fmt="%s",
               delimiter=",")
    np.savetxt(os.path.join(args.run_logdir,
                            "{}_picked_trainded_files_{}.csv".format(args.LOO_animal, len(valid_files))),
               np.array(trained_files), fmt="%s",
               delimiter=",")
    train_set = csv_reader_dataset3(trained_files, batch_size=batch_size, shuffle=False,
                          n_sec_per_sample=args.n_sec_per_sample, sr=args.sampling_rate)
    # valid_set = csv_reader_dataset3(valid_files, batch_size=batch_size,
    #                                shuffle=False, n_sec_per_sample=args.n_sec_per_sample, sr=512)
    # epg_set = csv_reader_dataset3(epg_files, batch_size=batch_size,
    #                              shuffle=False, n_sec_per_sample=args.n_sec_per_sample, sr=args.sampling_rate)

    total_num_batches_train = (len(trained_files) * 720 * 5) // (args.n_sec_per_sample * batch_size)
    total_num_batches_valid = (len(valid_files) * 720 * 5) // (args.n_sec_per_sample * batch_size)
    total_num_batches_epg = (len(epg_files) * 720 * 5) // (args.n_sec_per_sample * batch_size)
    
    hour_span_train = (get_timestamp_from_file(os.path.basename(trained_files[-1])) - get_timestamp_from_file(os.path.basename(trained_files[0]))) / 3600
    hour_span_valid = (get_timestamp_from_file(os.path.basename(valid_files[-1])) - get_timestamp_from_file(os.path.basename(valid_files[0]))) / 3600
    hour_span_epg = (get_timestamp_from_file(os.path.basename(epg_files[-1])) - get_timestamp_from_file(os.path.basename(epg_files[0]))) / 3600

    encoder = tf.keras.models.load_model(os.path.join(args.model_dir, 'encoder.h5'), compile=True)
    decoder = tf.keras.models.load_model(os.path.join(args.model_dir, 'decoder.h5'), compile=True)

    # disc_x = tf.keras.models.load_model(run_logdir+'/discriminator_x.h5')
    def compute_batch_distance(z):
        distance = []
        for i in range(z.shape[0]):
            distance.append(scipy.spatial.distance.euclidean(z[i].numpy(),
                                                             np.zeros(z_dim)))
        return np.array(distance)

    def compute_distros(dataset, directory, total_batch=10, num2coll=0, name="train"):
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
        all_filenames = []
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

            # all_filenames = np.vstack((all_filenames, batch_fn.numpy().reshape(-1,1)))
            all_filenames += list([[ele.decode("utf-8"), rat.decode("utf-8")] for ele, rat in
                 zip(batch_fn.numpy(), batch_rat_id.numpy())])
    
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
                np.savetxt(os.path.join(directory, "{}-collected_info-[fn,lb,id,err,dist,eeg,recon]-{}-outof-{}.csv".format(args.LOO_animal, i, total_batch)), np.array(coll_info), fmt="%s", delimiter=",")
                coll_info = 0
                
        np.save(directory + '/errors_{}.npy'.format(name), errors)
        # np.save(directory+'/probilities.npy', probilities)
        np.save(directory + '/distances_{}.npy'.format(name), distances)
        np.save(directory + '/z_{}.npy'.format(name), z_all[1:, :])
        np.savetxt(os.path.join(directory,
                                "all_filenames_{}.csv".format(name)), np.array(all_filenames),
                   fmt="%s", delimiter=",")
    
    
    ### TODO: add test function
    # test_files = []
    # rat_ids = []
    # for inds, batch_data in enumerate(valid_set):
    #     batch_features, batch_label, batch_fn, batch_rat_id = [batch_data[i]
    #                                                            for i in
    #                                                            range(
    #                                                                len(
    #                                                                    batch_data))]
    #     test_files += [ele.decode('utf-8') for ele in list(batch_fn.numpy())]
    #     rat_ids += [ele.decode('utf-8') for ele in list(batch_rat_id.numpy())]
        
    #
    compute_distros(train_set, output_directory + 'train_data', total_batch=total_num_batches_train, num2coll=total_num_batches_train//10, name="train")
    # compute_distros(valid_set, output_directory + 'valid_data', total_batch=total_num_batches_valid, num2coll=total_num_batches_valid//10, name="valid")
    # compute_distros(epg_set, output_directory + 'epg_data', total_batch=total_num_batches_epg, num2coll=total_num_batches_epg//10, name="epg")
    #
    ####################################################################
    # Load values from saved files
    # t_distances, t_errors = load_all_errors_and_distances(output_directory, name="train")
    # v_distances, v_errors = load_all_errors_and_distances(output_directory, name="valid")
    # e_distances, e_errors = load_all_errors_and_distances(output_directory, name="epg")
    #
    # ## plot values together for overview
    # plot_train_val_epg_values_together([t_errors,v_errors,e_errors], ["train", "valid", "epg"], output_directory, value_name="errors")
    # plot_train_val_epg_values_together([t_distances,v_distances,e_distances], ["train", "valid", "epg"], output_directory, value_name="distances")
    #
    # #### quantile analysis
    # whole_segment_t_errors = quantile_analysis_on_errors(t_errors, output_directory,
    #                                                      hlines_loc=None,
    #                                                      name="train")
    # th90 = np.percentile(whole_segment_t_errors, 90)  ## get the percentiles from the trained data
    # th95 = np.percentile(whole_segment_t_errors, 95)
    # th99_err = np.percentile(whole_segment_t_errors, 99)
    # whole_segment_e_errors = quantile_analysis_on_errors(e_errors, output_directory,
    #                                                      hlines_loc=[th90, th95,
    #                                                                  th99_err],
    #                                                      name="epg")
    # whole_segment_v_errors = quantile_analysis_on_errors(v_errors, output_directory,
    #                                                      hlines_loc=[th90, th95,
    #                                                                  th99_err],
    #                                                      name="valid")
    #
    # plot_train_val_epg_values_together([whole_segment_t_errors,
    #                                     whole_segment_v_errors,
    #                                     whole_segment_e_errors],
    #                                    ["train", "valid", "epg"],
    #                                    output_directory, value_name="whole segment error")
    #
    # ## plot percentile plots of reconstructions errors
    # plot_scatter_99th_quantile_error_segs(whole_segment_t_errors, th99_err, output_directory, name="train")
    # plot_scatter_99th_quantile_error_segs(whole_segment_v_errors, th99_err, output_directory, name="valid")
    # plot_scatter_99th_quantile_error_segs(whole_segment_e_errors, th99_err, output_directory, name="epg")
    #
    # th90_d = np.percentile(t_distances, 90)
    # th95_d = np.percentile(t_distances, 95)
    # th99_d = np.percentile(t_distances, 99)
    #
    # ## plot moving average of errors
    # moving_average_error_analysis(t_distances, output_directory, hlines_loc=None, name="train")
    # moving_average_error_analysis(v_distances, output_directory, hlines_loc=[th90_d, th95_d, th99_d], name="valid")
    # moving_average_error_analysis(e_distances, output_directory, hlines_loc=[th90_d, th95_d, th99_d], name="epg")
    #
    #
    # # pca = PCA(n_components=2)
    # # pca.fit(e_z_all)
    # # transformed_e = pca.transform(e_z_all)
    # # fig = plt.figure(figsize=(20,20))
    # # plt.scatter(transformed_e[:, 0], transformed_e[:,1], cmap='jet', c=whole_segment_e_errors, s=2)
    # # plt.xlabel('PC1')
    # # plt.ylabel('PC2')
    # # plt.legend()
    # # plt.colorbar()
    # # plt.show()
    # # plt.savefig(output_directory+'pca_z_e_color.png')
    # # plt.close()
    #
    #
    # test_window_in_minutes = [1, 5, 10, 15, 30, 60]
    # for window_in_minutes in test_window_in_minutes:
    #     window = int((window_in_minutes * 60) / 5)
    #     ## r = np.reshape(th_whole_segment_t_errors[:len(th_whole_segment_t_errors)//window*window], (-1, window))
    #     ## frequency_t = np.sum(np.where(r>0, 1, 0),axis=1)
    #     r = np.where(whole_segment_t_errors > th99_err, 1, 0)
    #     frequency_t = pd.Series(r).rolling(window).sum()
    #
    #     th99_frequency = np.nanpercentile(frequency_t, 99)
    #
    #     suprathreshold_freq_analysis(whole_segment_t_errors, output_directory, th99_err, th99_frequency,
    #                                  window, window_in_minutes, name="train")
    #     suprathreshold_freq_analysis(whole_segment_v_errors, output_directory, th99_err, th99_frequency,
    #                                  window, window_in_minutes, name="valid")
    #     suprathreshold_freq_analysis(whole_segment_e_errors, output_directory, th99_err, th99_frequency,
    #                                  window, window_in_minutes, name="epg")


    #############################################################################


def plot_train_val_epg_values_together(values, names, output_directory, value_name="Errors", iflogy=False ):
    fig = plt.figure(figsize=(10, 10))
    for value, name in zip(values, names):
        sns.distplot(value, kde=False, norm_hist=True, label='{}'.format(name))
    if iflogy:
        plt.yscale('log')
    plt.legend()
    plt.title(value_name)
    plt.savefig(output_directory + '{}-{}.png'.format(names, value_name))
    plt.savefig(output_directory + '{}-{}.pdf'.format(names, value_name), format="pdf")
    plt.close()


def load_all_errors_and_distances(output_directory, name="train"):
    t_errors = np.load(output_directory + '{}_data'.format(name) + '/errors_{}.npy'.format(name))
    # t_probilities = np.load(output_directory+'train_data'+'/probilities.npy')
    t_distances = np.load(output_directory + '{}_data'.format(name) + '/distances_{}.npy'.format(name))
    # t_z_all = np.load(output_directory+'train_data'+'/z.npy')
    return t_distances, t_errors


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

