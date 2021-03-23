import matplotlib.pyplot as plt
import os
import pdb
import time
from datetime import datetime
import tensorflow as tf
import numpy as np
import yaml
from sklearn.manifold import TSNE


def plot_errors(errors, path, err='reconstruction'):
    fig = plt.figure(figsize=(15,10))
    plt.plot(errors)
        
    if err == 'pdf':
        # plt.ylim(0, 1.4e-20)
        plt.ylabel('PDF')
        # ticks = np.arange(0,errors.shape[0], 720/2)
        # plt.xticks(ticks, np.arange(0,len(ticks)/2, 0.5))        
    if err == 'reconstruction':
        plt.ylim(0, 1500)
        plt.ylabel('Error')
        ticks = np.arange(0,errors.shape[0], 1843200/2)
        plt.xticks(ticks, np.arange(0,len(ticks)/2, 0.5))
    
    plt.grid(True)
    plt.xlabel('Time in hours')
    plt.savefig(path)
    plt.close()

def plot_dict_loss(d, run_logdir):
    fig = plt.figure(figsize=(30,20))
    # fig.subplots_adjust(hspace=0.4, wspace=0.2)
    for i, key in enumerate([x for x in list(d.keys()) if not x.startswith('v_')]):
        ax = fig.add_subplot(4, 3, i+1)
        ax.plot(d[key], label=key, linewidth=2, color='blue')

        # ax.plot(d['v_'+key], label='v_'+key, linewidth=1,  linestyle='dashed', color='red')

        # if max(d[key] + d['v_'+key]) > 1:
        #     ax.set_ylim([0, 1])
        ax.legend()
    plt.savefig(run_logdir+'/losses.png')
    plt.savefig(run_logdir+"/losses.pdf", format="pdf")

def plot_loss(history, run_logdir):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.title('Loss')
    plt.plot(loss, label="train")
    plt.plot(val_loss, label="validation")
    plt.legend()
    plt.savefig(run_logdir+'/loss.png')
    plt.close('all')

def plot_latent_space(model, valid_set, run_logdir, epoch):

    codes = []
    for batch in valid_set:
        code = model.predict(batch)
        # code = model.predict(batch)[:,:,0]
        codes.append(code)

    codes = np.array(codes)
    codes_flattened = np.reshape(codes, (codes.shape[0]*codes.shape[1], codes.shape[2]))
    # codes_embedded = TSNE(n_components=2).fit_transform(codes_flattened)

    plt.scatter(codes_flattened[:, 0], codes_flattened[:, 1], s=2)
    plt.savefig(run_logdir+'/latent_space_'+str(epoch)+'.png')
    plt.close()


def get_run_logdir(root_logdir, animal, args):
    """
    Get the experiment log dir
    :param root_logdir:
    :param animal:
    :param args:
    :return:n_pps2use
    """
    time_str = '{0:%Y-%m-%dT%H-%M-%S}'.format(datetime.now())
    run_id = "run_dim_{}_loss_weights_EPG_anomaly_{}_pps{}h_ctrl{}h_LOO_{}".format(args.z_dim, time_str, args.n_pps2use, args.n_ctrl2use, animal)
    path = os.path.join(root_logdir, run_id)
    os.mkdir(path)
    return path

def predict_validation_samples(model, valid_set, no_samples=6):
    
    random_dataset = tf.data.experimental.sample_from_datasets([valid_set])

    original_data = []
    reconstructions = []

    for item in random_dataset.take(no_samples):
        # reconstruction = model.predict(tf.expand_dims(item[0][0], axis=0))[0,:,0]
        # original_data.append(tf.expand_dims(item[0][0], axis=0)[0,:,0].numpy())
        # pdb.set_trace()
        reconstruction = model.predict(tf.expand_dims(item[0], axis=0))
        original_data.append(item[0].numpy())
        reconstructions.append(reconstruction[0].numpy())
    
    return original_data, reconstructions

def sample_data(model, z_dim, run_logdir, norm_params, std, epoch, no_samples=10):
    weights = np.ones(len(norm_params), dtype=np.float64) / len(norm_params)
    mixture_idx = np.random.choice(len(weights), size=no_samples, replace=True, p=weights)
    z = tf.convert_to_tensor([np.random.normal(norm_params[idx], std, size=(z_dim,1)) for idx in mixture_idx], dtype=tf.float32)

    x = model(z).numpy()
    row, col = 8, 4
    fig, axes = plt.subplots(row, col, sharex=True, figsize=(15, 10))
    for i in range(row*col):
        axes[i // col, np.mod(i, col)].plot(x[i], c='black',
                                            label='generated_data', linewidth=2)
        if np.mod(i, col) > 0:
            plt.setp(axes[i // col, np.mod(i, col)].get_yticklabels(),
                     visible=False)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.01, wspace=0.01)
    plt.legend(loc='upper right', shadow=True)
    plt.savefig(run_logdir+'/generated_data_'+str(epoch)+'.png')
    plt.close('all')


def plot_samples(original_data, reconstructions, run_logdir, epoch):
    row, col = 8, 4
    fig, axes = plt.subplots(row, col, sharex=True, figsize=(15,10))
    for i in range(row*col):
        axes[i // col, np.mod(i, col)].plot(original_data[i], c='red', label='original',  linewidth=2)
        axes[i // col, np.mod(i, col)].plot(reconstructions[i], c='black', label='reconstructed',  linewidth=2)
        if np.mod(i, col) > 0:
            plt.setp(axes[i // col, np.mod(i, col)].get_yticklabels(),
                     visible=False)
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, hspace=0.01, wspace=0.01)
    plt.legend(loc='upper right', shadow=True)
    plt.savefig(run_logdir+'/valid_samples_plot_'+str(epoch)+'.png')
    plt.savefig(run_logdir+'/valid_samples_plot_'+str(epoch)+'.pdf', format="pdf")
    plt.close('all')


def save_results(history, model, valid_set, note, run_logdir, no_samples=6):

    plot_loss(history, run_logdir)

    model.save(run_logdir+'/the_model.h5')

    original_data, reconstructions = predict_validation_samples(model, valid_set, no_samples=no_samples)

    plot_samples(original_data, reconstructions, run_logdir, 0)

    with open(run_logdir+'/notes.txt', 'a') as f:
        f.write(note)


def KnuthMorrisPratt(text, pattern):
    
    '''Yields all starting positions of copies of the pattern in the text.
Calling conventions are similar to string.find, but its arguments can be
lists or iterators, not just strings, it returns all matches, not just
the first one, and it does not need the whole text in memory at once.
Whenever it yields, it will have read the text exactly up to and including
the match that caused the yield.'''

    # allow indexing into pattern and protect against change during yield
    pattern = list(pattern)

    # build table of shift amounts
    shifts = [1] * (len(pattern) + 1)
    shift = 1
    for pos in range(len(pattern)):
        while shift <= pos and pattern[pos] != pattern[pos-shift]:
            shift += shifts[pos-shift]
        shifts[pos+1] = shift

    # do the actual search
    startPos = 0
    matchLen = 0
    for c in text:
        while matchLen == len(pattern) or \
              matchLen >= 0 and pattern[matchLen] != c:
            startPos += shifts[matchLen]
            matchLen -= shifts[matchLen]
        matchLen += 1
        if matchLen == len(pattern):
            yield startPos


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)
    
    def save_yaml(self, save_file_name):
        with open(save_file_name, 'w') as outfile:
            yaml.dump(self.__dict__, outfile, default_flow_style=False)


def load_parameters(filename):
    with open(filename) as f:
        ym_dicts = yaml.load(f, Loader=yaml.FullLoader)
        args = Struct(**ym_dicts)
    return  args


def get_dirs_with_platform(platform):
    """
    Given platform to get the root directories
    :param platform:
    :return:
    """
    paths_platforms = {"laptop": {
        "pps_data_path": "C:/Users/LDY/Desktop/EPG/EPG_data/data/3d/PPS",
        "ctrl_data_path": "C:/Users/LDY/Desktop/EPG/EPG_data/data/3d/control",
        "root_logdir": "C:/Users/LDY/Desktop/EPG/EPG_data/results"
    },
        "FIAS": {
            "pps_data_path": "/home/epilepsy-data/data/PPS-rats-from-Sebastian/PPS-Rats",
            "ctrl_data_path": "/home/epilepsy-data/data/PPS-rats-from-Sebastian/Control-Rats",
            "root_logdir": "/home/epilepsy-data/data/PPS-rats-from-Sebastian/resultsl-7rats"
        },
        "Farahat": {
            "pps_data_path": '/home/farahat/Documents/data',
            "ctrl_data_path": '/home/farahat/Documents/data',
            # TODO: your control dir
            "root_logdir": '/home/farahat/Documents/my_logs'
        }
    }
    pps_data_path = paths_platforms[platform]["pps_data_path"]
    ctrl_data_path = paths_platforms[platform]["ctrl_data_path"]
    root_logdir = paths_platforms[platform]["root_logdir"]
    
    return pps_data_path, ctrl_data_path, root_logdir



def copy_save_all_files(args):
    """
    Copy and save all files related to model directory
    :param args:
    :return:
    """
    src_dir = '.'
    save_dir = os.path.join(args.run_logdir, 'src')
    if not os.path.exists(save_dir):  # if subfolder doesn't exist, should make the directory and then save file.
        os.makedirs(save_dir)
    req_extentions = ['py', 'yaml', "sh"]
    for filename in os.listdir(src_dir):
        exten = filename.split('.')[-1]
        if exten in req_extentions:
            src_file_name = os.path.join(src_dir, filename)
            target_file_name = os.path.join(save_dir, filename)
            with open(src_file_name, 'r') as file_src:
                with open(target_file_name, 'w') as file_dst:
                    for line in file_src:
                        file_dst.write(line)
    print('Done WithCopy File!')


def get_timestamp_from_file(fn, year_ind=1):
    """
    Get the abs. time stamp for a given file
    EPG-2014-10-10T05-51-0-filter-5s-720-new.csv
    :param fn:
    :param year_ind:
    :return:
    """
    year = np.int(fn.split("-")[year_ind])
    mon = np.int(fn.split("-")[year_ind+1])
    day = np.int(fn.split("-")[year_ind+2].split("T")[0])
    hour = np.int(fn.split("-")[year_ind+2].split("T")[1])
    min = np.int(fn.split("-")[year_ind+3])
    timestamp = datetime(year, mon, day, hour, min).timestamp()
    return timestamp