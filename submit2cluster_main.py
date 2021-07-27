import time
import os
from utils import load_parameters, get_dirs_with_platform, get_run_logdir
from scanning import scan_animals_with_pretrained_model

class ClusterQueue:
	def __init__(self, animal, logdir, memory, yaml_file):

		# output path for the experiment log
		self.cmd_slurm = "sbatch --job-name {} --mem {} --output {}/%N_%j.log  --error {}/%N_%j.log".format(animal, memory, logdir, logdir)
		self.cmd_slurm += " submit2cluster.sh --yaml_file {}".format(yaml_file)
		
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


############################################################################################3
if __name__ == "__main__":
	# TODO: get parameters based on the platform
	# for submitting jobs to cluster, load the yaml, only change a few parameters regarding if_to_cluster
	args = load_parameters("./parameters.yaml")
	
	platform = "FIAS" #"Lu"  #
	
	pps_data_path, ctrl_data_path, root_logdir, src_dir = get_dirs_with_platform(platform)
	if platform == "FIAS":
		args.if_to_cluster = True #  False  #
	else:
		args.if_to_cluster = False  #True #
	
	args.platform = platform
	args.pps_data_path = pps_data_path
	args.ctrl_data_path = ctrl_data_path
	args.src_dir = src_dir
	args.root_logdir = root_logdir
	
	if args.platform == "Lu":
		args.LOO_animals = ["1227", "1275"]  # comment the rest on laptop
		args.pps_animals = ['1227', '1275']  # comment the rest on laptop
		args.ctrl_animals = ['3263', '3267'] # comment the rest on laptop
	
	# want to make each LOO training/scanning testing as one job
	if not args.if_scanning:  # this case is pure training
		for LOO_animal in args.LOO_animals:
			# create yaml file in src folder, later save it to each individual logdir
			# create output dir
			# please change platform in parameters.yaml file every time you change the running platform
			run_logdir = get_run_logdir(root_logdir, LOO_animal, args)
			
			args.run_logdir = run_logdir
			args.LOO_animal = LOO_animal
			
			# Generate new yaml file for train_aae.py, only changed LOO_animal
			yaml_filename = os.path.join(args.run_logdir, "{}_parameters.yaml".format(LOO_animal))
			args.save_yaml(yaml_filename)
			
			if args.if_to_cluster:
				ClusterQueue(LOO_animal, run_logdir, 10000, yaml_filename)
			else:
				print(
					"1. run this script to generate yaml file. 2. give the dir of the yaml file to the argument of train_aee.py")

	else:  # this is for scanning
		# list of the pretrained models
		models = [
			# r"C:\Users\LDY\Desktop\1-all-experiment-results\PPS-anomaly-detection\2021.07.21\from_me\run_dim_128_loss_weights_EPG_anomaly_2021-07-21T00-27-53_pps50h_ctrl100h_LOO_1227",
			"/home/epilepsy-data/data/PPS-rats-from-Sebastian/amr_logs/final_128_0.1/run_EPG_anomaly_2021_03_10-12_56_LOO_16_1275",
			"/home/epilepsy-data/data/PPS-rats-from-Sebastian/amr_logs/final_128_0.1/run_EPG_anomaly_2021_03_10-12_57_LOO_22_32140",
			"/home/epilepsy-data/data/PPS-rats-from-Sebastian/amr_logs/final_128_0.1/run_EPG_anomaly_2021_03_10-12_57_LOO_53_3263",
			# "/home/epilepsy-data/data/PPS-rats-from-Sebastian/amr_logs/final_128_0.1/run_EPG_anomaly_2021_03_10-13_11_LOO_33_1227",
			# "/home/epilepsy-data/data/PPS-rats-from-Sebastian/amr_logs/final_128_0.1/run_EPG_anomaly_2021_03_10-22_45_LOO_02_3266",
			# "/home/epilepsy-data/data/PPS-rats-from-Sebastian/amr_logs/final_128_0.1/run_EPG_anomaly_2021_03_11-00_12_LOO_23_1276",
			# "/home/epilepsy-data/data/PPS-rats-from-Sebastian/amr_logs/final_128_0.1/run_EPG_anomaly_2021_03_11-01_45_LOO_00_1237",
			# "/home/epilepsy-data/data/PPS-rats-from-Sebastian/amr_logs/final_128_0.1/run_EPG_anomaly_2021_03_11-02_08_LOO_25_32141",
			# "/home/epilepsy-data/data/PPS-rats-from-Sebastian/amr_logs/final_128_0.1/run_EPG_anomaly_2021_03_11-10_20_LOO_09_3267",
			# "/home/epilepsy-data/data/PPS-rats-from-Sebastian/amr_logs/final_128_0.1/run_EPG_anomaly_2021_03_12-16_42_LOO_18_1270",
		]
		LOO_animals = [os.path.basename(dd).split("_")[-1] for dd in models]

		for LOO_animal, model_dir in zip(LOO_animals, models):
			args.LOO_animal = LOO_animal
			args.model_dir = model_dir
			args.run_logdir = os.path.join(args.root_logdir, os.path.basename(model_dir)+"-scanning")
			if not os.path.exists(args.run_logdir):
				os.makedirs(args.run_logdir)

			# Generate new yaml file for train_aae.py, only changed LOO_animal
			yaml_filename = os.path.join(args.run_logdir,
			                             "{}_scanning_parameters.yaml".format(
				                             LOO_animal))
			args.save_yaml(yaml_filename)
			
			if args.if_to_cluster:
				ClusterQueue("scan"+args.LOO_animal, args.run_logdir, 12000, yaml_filename)
			else:
				print("1. run this script to generate yaml file. 2. give the dir of the yaml file to the argument of train_aee.py")
			
	#
	# # #
	# # # # Want to use for loop in the train_aae.py
	# else:
	#
	# """
	# """
	# 	for LOO_animal in args.LOO_animals:
	# 		# create yaml file in src folder, later save it to each individual logdir
	# 		# create output dir
	# 		run_logdir = get_run_logdir(root_logdir, LOO_animal, args)
	#
	# 		cmd_slurm = "srun -p sleuths -w jetski --mem=20000 --job-name EPG --output {}/%N_%j.log --error {}/%N_%j.log --gres gpu:rtx2080ti:1 python3 train_aae.py ".format(
	# 		args.run_logdir)
	#
	# 	os.system(cmd_slurm)

