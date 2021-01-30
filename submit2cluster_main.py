import time
import os


class ClusterQueue:
	def __init__(self, dirs):
		self.output_path = dirs
		
		dir_root = os.path.basename(os.path.dirname(self.output_path)).split("-")
		
		# output path for the experiment log
		self.cmd_slurm = "sbatch --job-name EPG --output {}/%N_%j.log".format(
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


############################################################################################3
if __name__ == "__main__":
	# Creating the flags to be passed to classifier.py
	# Get all parameters and generate the output folders
	
	if_to_cluster = False
	# want to make one LOO training as one job, or  one scanning testing as one job
	if if_to_cluster:
		## two cases to train: parallel with cluster, so there won't be a loop of LOO_animals
		for dirs in args.all_run_logdirs:
			ClusterQueue(dirs)
	
	# Want to use for loop in the train_aae.py
	else:
		cmd_slurm = "srun -p sleuths -w jetski --mem=20000 --job-name EPG --output {}/%N_%j.log --error {}/%N_%j.log --gres gpu:rtx2080ti:1 python3 train_aae.py ".format(
			args.run_logdir)
		
		os.system(cmd_slurm)

