import tensorflow as tf
import layer

class net(layer.net):

	def addConvLayer(self, nChannels, nOutChannels, dropRate):
		concate = nn.Concat(2)
		concate.add(nn.Identity())  # ?
		convFactory.add(tf.nn.SpatialBatchNormalization(nChannels))
		convFactory.add(tf.nn.ReLU(true))
		convFactory.add(tf.nn.SpatialConvolution(nChannels, nOutChannels, 3, 3, 1, 1, 1, 1))
		if dropRate: convFactory.add(nn.Dropout(dropRate))
		concate.add(convFactory)
		self.add(concate)

	def addTransition(self, nChannels, nOutChannels, dropRate):
		self.add(tf.nn.SpatialBatchNormalization(nChannels))
		self.add(tf.nn.ReLU(true))
		self.add(tf.nn.SpatialConvolution(nChannels, nOutChannels, 1, 1, 1, 1, 0, 0))
		if dropRate: self.add(nn.Dropout(dropRate))
		self.add(tf.nn.SpatialAveragePooling(2, 2))
		self.conv([1,1,1,1],pool=True,dropout=dropRate) # ja?


	def buildDenseConv(self):
		depth = 3 * 1 + 4
		if (depth - 4) % 3 == 0:  raise ("Depth must be 3N + 4! (4,7,10,...) ")  # # layers in each denseblock
		N = (depth - 4) / 3
		dropRate = nil  # nil to disable dropout, non - zero number to enable dropout and set drop rate
		# # channels before entering the first denseblock ??
		# set it to be comparable with growth rate ??
		nChannels = 16
		growthRate = 12

		self.add(tf.nn.SpatialConvolution(3, nChannels, 3, 3, 1, 1, 1, 1))

		for i in range(N):
			addConvLayer(self, nChannels, growthRate, dropRate)
			nChannels = nChannels + growthRate
			addTransition(self, nChannels, nChannels, dropRate)

		for i in range(N):
			addConvLayer(self, nChannels, growthRate, dropRate)
			nChannels = nChannels + growthRate
			addTransition(self, nChannels, nChannels, dropRate)

		for i in range(N):
			addConvLayer(self, nChannels, growthRate, dropRate)
			nChannels = nChannels + growthRate

		self.add(tf.nn.SpatialBatchNormalization(nChannels))
		self.add(tf.nn.ReLU(true))
		self.add(tf.nn.SpatialAveragePooling(8, 8)).add(nn.Reshape(nChannels))
		if opt.dataset == 'cifar100':
			self.add(nn.Linear(nChannels, 100))
		elif opt.dataset == 'cifar10':
			self.add(nn.Linear(nChannels, 10))
		else:
			raise ("Dataset not supported yet!")
