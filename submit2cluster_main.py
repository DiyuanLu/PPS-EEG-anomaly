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
	cmds_to_sh = []
	for config_files in config_dirs:  # three arguments
		cmd_python = ""
		for k, v in zip(["output_path", "exp_config", "model_config"],
		                [config_files[0], config_files[1], config_files[2]]):
			# _key_to_flag transforms "something_stupid"   into   "--something-stupid"
			flag = _key_to_flag(k)
			# _to_arg transforms ("--something-stupid", a_value)   into   "--something-stupid a_value"
			arg = _to_arg(flag, v)
			# arg = _to_arg(flag, os.path.join(root_exp, v))
			cmd_python += arg
		cmds_to_sh.append(cmd_python)
		# cmds_to_sh.append(cmd_python + " --output {}/%N_%j.log".format(config_files[0]))
		# cmd_python = "" --output {}/%N_%j.log".format(config_files[0])"
	
	commands = ''
	for cmds in cmds_to_sh:
		commands += "\"{}\" ".format(cmds)
	

	for dirs in config_dirs:
		ClusterQueue(dirs)
	

