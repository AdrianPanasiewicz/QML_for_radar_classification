from torch import nn

class ClassicalNeuralNetwork(nn.Module):
	def __init__(self, l1=500, l2=250):
		super().__init__()
		self.init_kwargs = {"l1": l1, "l2": l2}
		self.flatten = nn.Flatten()
		self.linear_relu_stack = nn.Sequential(
			nn.Linear(10, l1),
			nn.ReLU(),
			nn.Linear(l1, l2),
			nn.ReLU(),
			nn.Linear(l2, 100),
			nn.ReLU(),
			nn.Linear(100, 3),
		)

	def forward(self, x):
		x = self.flatten(x)
		logits = self.linear_relu_stack(x)
		return logits
