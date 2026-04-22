from torch import nn

class ClassicalNeuralNetwork(nn.Module):
	def __init__(self, model_config):
		super().__init__()
		self.init_kwargs = {"layers": model_config['layers'], "neurons_per_layer": model_config['neurons_per_layer']}
		self.model_name = self.__class__.__name__
		# self.flatten = nn.Flatten()
		layers_list = [nn.Linear(10, self.init_kwargs['neurons_per_layer']), nn.ReLU()]
		for _ in range(self.init_kwargs['layers']):
			layers_list.append(nn.Linear(self.init_kwargs['neurons_per_layer'], self.init_kwargs['neurons_per_layer']))
			layers_list.append(nn.LeakyReLU(0.01))
		layers_list.append(nn.Linear(self.init_kwargs['neurons_per_layer'], 2))
		self.linear_relu_stack = nn.Sequential(*layers_list)

	def forward(self, x):
		# x = self.flatten(x)
		logits = self.linear_relu_stack(x)
		return logits
