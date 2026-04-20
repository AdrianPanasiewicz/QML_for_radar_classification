from torch import nn

class ClassicalNeuralNetwork(nn.Module):
	def __init__(self, layers, neurons_per_layer):
		super().__init__()
		self.init_kwargs = {"layers": layers, "neurons_per_layer": neurons_per_layer}
		self.model_name = self.__class__.__name__
		# self.flatten = nn.Flatten()
		layers_list = [nn.Linear(10, neurons_per_layer), nn.ReLU()]
		for _ in range(layers):
			layers_list.append(nn.Linear(neurons_per_layer, neurons_per_layer))
			layers_list.append(nn.LeakyReLU(0.01))
		layers_list.append(nn.Linear(neurons_per_layer, 2))
		self.linear_relu_stack = nn.Sequential(*layers_list)

	def forward(self, x):
		# x = self.flatten(x)
		logits = self.linear_relu_stack(x)
		return logits
