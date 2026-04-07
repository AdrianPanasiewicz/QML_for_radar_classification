from torch import nn

class ClassicalSupportVectorMachine(nn.Module):
	def __init__(self):
		super().__init__()
		self.init_kwargs = {}
		self.model_name = self.__class__.__name__
		self.flatten = nn.Flatten()
		self.linear_relu_stack = nn.Sequential(
			nn.Linear(2*16*64, 500),
			nn.ReLU(),
			nn.Linear(500, 250),
			nn.ReLU(),
			nn.Linear(250, 100),
			nn.ReLU(),
			nn.Linear(100, 6),
		)

	def forward(self, x):
		x = self.flatten(x)
		logits = self.linear_relu_stack(x)
		return logits
