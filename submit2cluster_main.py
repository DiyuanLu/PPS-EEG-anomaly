import time
import os
import yaml
from utils import load_parameters, get_dirs_with_platform, get_run_logdir

class ClusterQueue:
	def __init__(self, animal, logdir):

		# output path for the experiment log
		self.cmd_slurm = "sbatch --job-name EPG{} --output {}/%N_%j.log".format(animal,
		                                                                        logdir)
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
		
	# get parameters
	platform = "Lu_laptop"
	pps_data_path, ctrl_data_path, root_logdir = get_dirs_with_platform(platform)

	if_to_cluster = True
	
	# want to make one LOO training as one job, or  one scanning testing as one job
	if if_to_cluster:
		# load yaml file
		args = load_parameters("./parameters.yaml")
		## Generate output logdirs for all experiements
		all_run_logdirs = []
		for LOO_animal in args.LOO_animals:
			# create yaml file in src folder, later save it to each individual logdir
			# create output dir
			run_logdir = get_run_logdir(root_logdir, LOO_animal, args)

			ClusterQueue(LOO_animal, run_logdir)
	
	# Want to use for loop in the train_aae.py
	else:
		cmd_slurm = "srun -p sleuths -w jetski --mem=20000 --job-name EPG --output {}/%N_%j.log --error {}/%N_%j.log --gres gpu:rtx2080ti:1 python3 train_aae.py ".format(
			args.run_logdir)
		
		os.system(cmd_slurm)

