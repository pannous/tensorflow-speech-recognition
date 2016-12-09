import os
import subprocess  # NEW WAY!

tensorboard_logs = '/tmp/tensorboard_logs/'
global logdir

def get_last_tensorboard_run_nr():
	if not os.path.exists(tensorboard_logs ):
		os.system("mkdir " + tensorboard_logs )
		return 0
	logs=subprocess.check_output(["ls", tensorboard_logs]).split("\n")
	# print("logs: ",logs)
	runs=map(lambda x: (not x.startswith("run") and -1) or int(x[-1]) ,logs)
	# print("runs ",runs)
	return max(runs)+1



def set_tensorboard_run(reset=False,auto_increment=True,run_nr=-1):
	if run_nr < 1 or auto_increment:
		run_nr = get_last_tensorboard_run_nr()
	if run_nr == 0 or reset:
		run_nr=0
		clear_tensorboard()
	print("RUN NUMBER " + str(run_nr))
	global logdir
	if run_nr>0 and len(os.listdir(tensorboard_logs + 'run' + str(run_nr-1)))==0:
		run_nr=run_nr-1 #   previous run was not successful

	logdir = tensorboard_logs + 'run' + str(run_nr)
	if not os.path.exists(logdir):
		os.system("mkdir " + logdir)


def clear_tensorboard():
	os.system("rm -rf %s/*" % tensorboard_logs)  # sync

def nop():
	return tf.constant("nop")
	# pass

def show_tensorboard():
		print("run: tensorboard --debug --logdir=" + tensorboard_logs+" and navigate to http://0.0.0.0:6006")

def kill_tensorboard():
	os.system("ps -afx | grep tensorboard | grep -v 'grep' | awk '{print $2}'| xargs kill -9")

def current_logdir():
	print("current logdir: "+logdir)
	return logdir

def run_tensorboard(restart=False,show_browser=False):
	if restart: kill_tensorboard()
	subprocess.Popen(["tensorboard", '--logdir=' + tensorboard_logs])  # async
	# os.system("sleep 5; open http://0.0.0.0:6006")
	if(show_browser):
		subprocess.Popen(["open", 'http://0.0.0.0:6006'])  # async

# run_tensorboard()
