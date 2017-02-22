#!/usr/bin/env python
import numpy

import record
import layer

frame = record.next_frame() # Generator

def baseline():
	pass

# net=layer.net(baseline)
net = layer.identity()

def error(out, data):
	return numpy.sum(numpy.abs(out-data))/len(data)


if __name__ == '__main__':
	prediction=0
	while 1:
		data= next(frame)
		print(error(prediction, data))
		prediction=net.predict(data)



