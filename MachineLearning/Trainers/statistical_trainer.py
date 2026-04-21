from pathlib import Path

import torch
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from MachineLearning.Trainers.abstract_trainer import AbstractTrainer
from MachineLearning.Models.experiment_pure.classical_neural_network import ClassicalNeuralNetwork

MODEL_REGISTRY = {
    "ClassicalNeuralNetwork": ClassicalNeuralNetwork,
}


class TrainerForModelStatistics(AbstractTrainer):
    def __init__(self, training_path, validating_path, testing_path, criterion):
        super().__init__(training_path, validating_path, testing_path, criterion)

    def return_config_from_ray_results(self, best_result, map_location="cpu"):
        checkpoint = best_result.checkpoint
        with checkpoint.as_directory() as checkpoint_dir:
            checkpoint_path = Path(checkpoint_dir) / "checkpoint.pt"
            checkpoint_data = torch.load(checkpoint_path, map_location=map_location)

        model_class_name = checkpoint_data["model_class_name"]
        model_init_kwargs = checkpoint_data["model_init_kwargs"]
        net_state_dict = checkpoint_data["net_state_dict"]

        # model_class = MODEL_REGISTRY[model_class_name]
        # model = model_class(**model_init_kwargs)
        # model.load_state_dict(net_state_dict)
        # model.eval()
        return model_class_name, model_init_kwargs, net_state_dict


    def train_model(self, model_class, config, number_of_trials=10, number_of_epochs=50):

        device = config["device"]
        data_array_all_runs = []

        for model_run in tqdm(range(number_of_trials), desc="Model runs"):

            net = model_class(
                layers=config["layers"],
                neurons_per_layer=config["neurons_per_layer"]
            ).to(device)

            optimizer = Adam(net.parameters(), lr=config["lr"])
            # optimizer = SGD(net.parameters(), lr=config["lr"], momentum=0.9) # Do poprawienia
            trainloader = DataLoader(self.trainset, batch_size=int(config["batch_size"]), shuffle=True, num_workers=2) # Do poprawienia
            valloader = DataLoader(self.valset, batch_size=int(config["batch_size"]), shuffle=False, num_workers=2)

            data_dict_per_epoch = {
                "accuracy": [],
                "validation_loss": []
            }

            for epoch in tqdm(range(number_of_epochs), desc='Epochs'):
                net.train()

                for inputs, labels in trainloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = net(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                net.eval()
                val_loss = 0.0
                val_steps = 0
                total = 0
                correct = 0

                with torch.no_grad():
                    for inputs, labels in valloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = net(inputs)
                        loss = self.criterion(outputs, labels)
                        val_loss += loss.item()
                        val_steps += 1
                        predicted = outputs.argmax(dim=1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()


                data_dict_per_epoch["validation_loss"].append(val_loss / val_steps)
                data_dict_per_epoch["accuracy"].append(100 * correct / total)

            data_array_all_runs.append(data_dict_per_epoch)

        return data_array_all_runs

    def test_model(self, model):
        pass