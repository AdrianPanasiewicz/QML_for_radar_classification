from pathlib import Path

import os
import gc
import torch
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from MachineLearning.Trainers.abstract_trainer import AbstractTrainer
from MachineLearning.Models.experiment_pure.classical_neural_network import ClassicalNeuralNetwork
from MachineLearning.Models.experiment_pure.quantum_neural_network import QuantumNeuralNetwork


class StatisticalTrainer(AbstractTrainer):
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

        metrics_dictionary = {
            "training_data": list(),
            "testing_data": list(),
        }

        trainloader = DataLoader(self.trainset,
                                 batch_size=int(training_config["batch_size"]),
                                 shuffle=True,
                                 num_workers=training_config["number_of_training_workers"],
                                 pin_memory=False if training_config["device"]=="cpu" else True,
                                 persistent_workers=True
                                 )
        valloader = DataLoader(self.valset,
                               batch_size=int(training_config["batch_size"]),
                               shuffle=False,
                               num_workers=training_config["number_of_validating_workers"],
                               pin_memory=False if training_config["device"]=="cpu" else True,
                               persistent_workers=True
                               )

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
                    outputs = net(inputs).squeeze()

                    if isinstance(net, QuantumNeuralNetwork):
                        probs = torch.clamp((outputs + 1.0) / 2.0, 0.0, 1.0)
                        loss = self.criterion(probs, labels.float())
                    else:
                        loss = self.criterion(outputs, labels.long())

                    if training_config["regularization"]["type"] is not None:
                        if training_config["regularization"]['type'] == 'l1':
                            penality = sum(p.abs().sum() for p in net.parameters())
                            loss = loss + training_config["regularization"]["lambda"] * penality

                        elif training_config['regularization']['type'] == 'l2':
                            penality = sum((p ** 2).sum() for p in net.parameters())
                            loss = loss + training_config["regularization"]["lambda"] * penality

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
                            probs = torch.clamp((outputs + 1.0) / 2.0, 0.0, 1.0)
                            loss = self.criterion(probs, labels.float())
                            predicted = (outputs >= 0).long()

                        else:
                            outputs = net(inputs)
                            loss = self.criterion(outputs, labels.long())
                            predicted = outputs.argmax(dim=1)

                        if training_config["regularization"]["type"] is not None:
                            if training_config["regularization"]['type'] == 'l1':
                                penality = sum(p.abs().sum() for p in net.parameters())
                                loss = loss + training_config["regularization"]["lambda"] * penality

                            elif training_config['regularization']['type'] == 'l2':
                                penality = sum((p ** 2).sum() for p in net.parameters())
                                loss = loss + training_config["regularization"]["lambda"] * penality

                        val_loss += loss.item()
                        val_steps += 1

                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()


                data_dict_per_epoch["validation_loss"].append(val_loss / val_steps)
                data_dict_per_epoch["accuracy"].append(100 * correct / total)

            metrics = self.test_model(net, training_config["device"], training_config["number_of_testing_workers"])

            metrics_dictionary["training_data"].append(data_dict_per_epoch)
            metrics_dictionary["testing_data"].append(metrics)

            gc.collect()

        del trainloader
        del valloader

        return net, metrics_dictionary

    def test_model(self, model, device="cpu", num_workers=2):

        model = model.to(device)
        testloader = DataLoader(
            self.testset,
            num_workers=num_workers
        )

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                if isinstance(model, QuantumNeuralNetwork):
                    preds = (outputs >= 0).long()
                else:
                    _, preds = torch.max(outputs.data, 1)

                all_preds.append(preds)
                all_labels.append(labels)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        metrics = self.calculate_metrics(all_labels,all_preds)

        return metrics

    def calculate_metrics(self, labels, predictions):

        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        accuracy = accuracy_score(labels, predictions)
        balanced_acc = balanced_accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        f1 = f1_score(labels, predictions, zero_division=0)

        return {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}
        }