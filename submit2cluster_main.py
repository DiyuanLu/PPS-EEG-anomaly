import time
import os
from utils import load_parameters, get_dirs_with_platform, get_run_logdir

class ClusterQueue:
	def __init__(self, animal, logdir, memory, yaml_file):

		# output path for the experiment log
		self.cmd_slurm = "sbatch --job-name EPG-{} --mem {} --output {}/%N_%j.log  --error {}/%N_%j.log".format(animal, memory, logdir, logdir)
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
	# Creating the flags to be passed to classifier.py
	# Get all parameters and generate the output folders
	dict_file = [{'sports': ['soccer', 'football', 'basketball', 'cricket',
	                         'hockey', 'table tennis']},
	             {'countries': ['Pakistan', 'USA', 'India', 'China', 'Germany',
	                            'France', 'Spain']}]
	
	# with open(r'E:\data\store_file.yaml', 'w') as file:
	# 	documents = yaml.dump(dict_file, file)
		
	# TODO: get parameters based on the platform
	platform = "FIAS"
	pps_data_path, ctrl_data_path, root_logdir = get_dirs_with_platform(platform)
	if_to_cluster = True
	
	# for submitting jobs to cluster, load the yaml, only change a few parameters regarding if_to_cluster
	args = load_parameters("./parameters.yaml")
	args.platform = platform
	
	# want to make one LOO training as one job, or  one scanning testing as one job
	if if_to_cluster:
		# load yaml file
		args.if_to_cluster = True
		if not args.if_scanning:
			for LOO_animal in args.LOO_animals:
				# create yaml file in src folder, later save it to each individual logdir
				# create output dir
				run_logdir = get_run_logdir(root_logdir, LOO_animal, args)
				args.pps_data_path = pps_data_path
				args.ctrl_data_path = ctrl_data_path
				args.root_logdir = root_logdir
				
				args.run_logdir = run_logdir
				args.LOO_animal = LOO_animal
				
				# Generate new yaml file for train_aae.py, only changed LOO_animal
				yaml_filename = os.path.join(run_logdir, "{}_parameters.yaml".format(LOO_animal))
				args.save_yaml(yaml_filename)
				
				ClusterQueue(LOO_animal, run_logdir, 15000, yaml_filename)
		else:
			models = [
				"/home/epilepsy-data/data/PPS-rats-from-Sebastian/resultsl-7rats/run_dim_16_EPG_anomaly_2021-01-27T06-07-51_pps20h_ctrl100h_LOO_1275"
			]
			LOO_animals = [os.path.basename(dd).split("_")[-1] for dd in models]
			args.pps_data_path = pps_data_path
			args.ctrl_data_path = ctrl_data_path
			args.root_logdir = root_logdir
			for LOO_animal, model_dir in zip(LOO_animals, models):
				args.LOO_animal = LOO_animal
				args.model_dir = model_dir
				args.LOO_animal = LOO_animal
				# Generate new yaml file for train_aae.py, only changed LOO_animal
				yaml_filename = os.path.join(model_dir,
				                             "{}_scanning_parameters.yaml".format(
					                             LOO_animal))
				args.save_yaml(yaml_filename)
				
				ClusterQueue("scan"+args.LOO_animal, model_dir, 12000, yaml_filename)
			
	#
	# # Want to use for loop in the train_aae.py
	# else:
	# 	for LOO_animal in args.LOO_animals:
	# 		# create yaml file in src folder, later save it to each individual logdir
	# 		# create output dir
	# 		run_logdir = get_run_logdir(root_logdir, LOO_animal, args)
	#
	# 		cmd_slurm = "srun -p sleuths -w jetski --mem=20000 --job-name EPG --output {}/%N_%j.log --error {}/%N_%j.log --gres gpu:rtx2080ti:1 python3 train_aae.py ".format(
	# 		args.run_logdir)
	#
	# 	os.system(cmd_slurm)

