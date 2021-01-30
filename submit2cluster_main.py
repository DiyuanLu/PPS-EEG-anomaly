import time
import os


class ClusterQueue:
	def __init__(self, animal, dirs):
		self.output_path = dirs
		
		dir_root = os.path.basename(os.path.dirname(self.output_path)).split("-")
		
		# output path for the experiment log
		self.cmd_slurm = "sbatch --job-name EPG{} --output {}/%N_%j.log".format(animal,
			self.output_path)
		
		# special treatment for the "description" param (for convevience)
		# self.cmd_slurm += " --job-name {}".format(kwargs["description"])
		self.cmd_slurm += " submit2cluster.sh"
		
		print("#########################################################")
		print(self.cmd_slurm, "\n")
		print("##########################################################\n")
		# TODO
		os.system(self.cmd_slurm)
		
		time.sleep(1)
	
	def _key_to_flag(self, key):
		return "--" + key.replace("_", "_")
	
	def _to_arg(self, flag, v):
		return " {} {}".format(flag, v)
	
	def watch_tail(self):
		os.system(
			"watch tail -n 40 \"{}\"".format(self.output_path + "/log/*.log"))


def _key_to_flag(key):
	return "--" + key.replace("_", "_")


def _to_arg(flag, v):
	return " {} {}".format(flag, v)


def get_parameters(platform):
	paths_platforms = {"Lu_laptop": {
		"pps_data_path": "C:/Users/LDY/Desktop/EPG/EPG_data/data/3d/PPS",
		"ctrl_data_path": "C:/Users/LDY/Desktop/EPG/EPG_data/data/3d/control",
		"root_logdir": "C:/Users/LDY/Desktop/EPG/EPG_data/results"
		},
	                   "FIAS_cluster": {
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
	#
	parameters = {
		"LOO": True,  # False
		"if_scanning": True,
		"n_sec_per_sample": 1,
		"sampling_rate": 512,
		"input_size": 512,
		"h_dim": 512,
		"z_dim": 16,
		"n_epochs": 100,
		"batch_size": 128,
		"LOO_animals": ["32141"],
		# "32140", , "1276" "1275", "1276",["1275", "1276", "32140", "32141"],
		"n_pps2use": 20,  # 20,
		"n_ctrl2use": 100,  # 100,
		"train_percentage": 0.9,
		"pps_animals": ["1227", "1237", "1270", "1275", "1276", "32140", "32141"],
		#
		"ctrl_animals": ["3263", "3266", "3267"],
		"file_pattern": "new.csv",
		"if_include_ctrl": True,  # whether to include ctrl animals
		
		# model related parameters
		"std": 0.1,
		"kernel_size": 5,
		"ae_loss_weight": 1.0,
		"reg_loss_weight": 0.0,
		"gen_z_loss_weight": 1.0,
		"gen_x_loss_weight": 0.0,
		"dc_loss_weight": 1.0,
		
		# data path related parameters
		"platform": platform,
		"pps_data_path": pps_data_path,
		"ctrl_data_path": ctrl_data_path,
		"root_logdir": root_logdir
	}
	
	args = Struct(**parameters)
	assert args.input_size == args.n_sec_per_sample * args.sampling_rate, "input size is wrong!"
	
	if args.platform == "Farahat":
		tf.enable_eager_execution()
	
	return args

############################################################################################3
if __name__ == "__main__":
	# Creating the flags to be passed to classifier.py
	# Get all parameters and generate the output folders
	
	# get parameters
	platform = "FIAS_cluster"
	args = get_parameters(platform)
	
	if_to_cluster = False
	# want to make one LOO training as one job, or  one scanning testing as one job
	if if_to_cluster:
		## two cases to train: parallel with cluster, so there won't be a loop of LOO_animals
		for LOO_animal, LOO_output_dir in zip(args.all_run_logdirs):
			#generate yaml file
			
			ClusterQueue(LOO_animal, LOO_output_dir, yaml_file)
	
	# Want to use for loop in the train_aae.py
	else:
		cmd_slurm = "srun -p sleuths -w jetski --mem=20000 --job-name EPG --output {}/%N_%j.log --error {}/%N_%j.log --gres gpu:rtx2080ti:1 python3 train_aae.py ".format(
			args.run_logdir)
		
		os.system(cmd_slurm)

