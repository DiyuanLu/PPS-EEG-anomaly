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
		if_to_cluster = True#  False  #
	else:
		if_to_cluster = False  #
	
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
	if if_to_cluster:
		args.if_to_cluster = True
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
				
				ClusterQueue(LOO_animal, run_logdir, 10000, yaml_filename)
		else:  # this is for scanning
			# list of the pretrained models
			models = [
				"/home/epilepsy-data/data/PPS-rats-from-Sebastian/results-7rats/2021-04-29T17-16-16-anomaly_BL_w_ctrl-VAE_MLP-LOO-all-rats/2021-04-27T17-16-16-anomaly_BL_w_ctrl-VAE_MLP-LOO-1227-dim_128-50h-train-1227",
			]
			LOO_animals = [os.path.basename(dd).split("_")[-1] for dd in models]

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

