from pathlib import Path

import os
import gc
import torch
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from MachineLearning.Trainers.abstract_trainer import AbstractTrainer
from MachineLearning.Models.experiment_pure.classical_neural_network import ClassicalNeuralNetwork
from MachineLearning.Models.experiment_pure.quantum_neural_network import QuantumNeuralNetwork

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


    def train_model(self, model_class, config):

        training_config = config["training_config"]
        model_config = config["model_config"]

        device = training_config["device"]
        data_array_all_runs = []

        trainloader = DataLoader(self.trainset,
                                 batch_size=int(training_config["batch_size"]),
                                 shuffle=True,
                                 num_workers=4,
                                 pin_memory=False if training_config["device"]=="cpu" else True,
                                 )
        valloader = DataLoader(self.valset,
                               batch_size=int(training_config["batch_size"]),
                               shuffle=False,
                               num_workers=2,
                               pin_memory=False if training_config["device"]=="cpu" else True,
                               )

        threads = 8
        torch.set_num_threads(threads)
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)

        for model_run in tqdm(range(training_config['number_of_trials']), desc="Model runs"):

            net = model_class(model_config).to(device)

            if training_config["optimizer"]["name"] == "SGD":
                optimizer = SGD(
                    net.parameters(),
                    lr=training_config["optimizer"]["lr"],
                    momentum=training_config["optimizer"]["momentum"],
                    weight_decay=training_config["optimizer"]["weight_decay"],
                )
            elif training_config["optimizer"]["name"] == "Adam":
                optimizer = Adam(
                    net.parameters(),
                    lr=training_config["optimizer"]["lr"],
                    weight_decay=training_config["optimizer"]["weight_decay"],
                )
            else:
                raise TypeError(
                    f"Assigned {training_config["optimizer"]["name"]} is not implemented in hyperparameter search")

            data_dict_per_epoch = {
                "accuracy": [],
                "validation_loss": []
            }

            for epoch in tqdm(range(training_config['epochs']), desc='Epochs'):
                net.train()

                for inputs, labels in trainloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    if isinstance(net, QuantumNeuralNetwork):
                        outputs = net(inputs).squeeze()
                        probs = (outputs + 1.0) / 2.0
                        loss = self.criterion(probs, labels.float())
                    else:
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
                        if isinstance(net, QuantumNeuralNetwork):
                            outputs = net(inputs).squeeze()
                            probs = (outputs + 1.0) / 2.0
                            loss = self.criterion(probs, labels.float())
                            predicted = (outputs >= 0).long()

                        else:
                            outputs = net(inputs)
                            loss = self.criterion(outputs, labels)
                            predicted = outputs.argmax(dim=1)

                        val_loss += loss.detach()
                        val_steps += 1

                        total += labels.size(0)
                        correct += (predicted == labels).sum().detach()


                data_dict_per_epoch["validation_loss"].append(val_loss / val_steps)
                data_dict_per_epoch["accuracy"].append(100 * correct / total)

            data_array_all_runs.append(data_dict_per_epoch)
            gc.collect()

        return data_array_all_runs

    def test_model(self, model):
        pass