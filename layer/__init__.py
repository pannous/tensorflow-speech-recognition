from net import *

# PyCharm horrible auto-complete if separated into different modules
# from conv import *
# from batch_norm import *
# from dense import *
# from densenet import *

__all__ = ["net"]


def clear_tensorboard():
	os.system("rm -rf /tmp/tensorboard_logs/*")  # sync
# clear_tensorboard()


def nop():
	return tf.constant("nop")
	# pass


def show_tensorboard():
		print("run: tensorboard --debug --logdir=" + tensorboard_logs+" and navigate to http://0.0.0.0:6006")

def run_tensorboard():
		import subprocess  # NEW WAY!
		subprocess.call(["tensorboard", '--logdir=' + tensorboard_logs])  # async
		print("OK")
		subprocess.call(["open", 'http://0.0.0.0:6006'])  # async

# run_tensorboard()
show_tensorboard()
