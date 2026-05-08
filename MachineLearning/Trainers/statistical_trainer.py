from pathlib import Path

import pickle
import os
import gc
import torch
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

from MachineLearning.Trainers.abstract_trainer import AbstractTrainer
from MachineLearning.Models.experiment_pure.classical_neural_network import ClassicalNeuralNetwork
from MachineLearning.Models.experiment_pure.quantum_neural_network import QuantumNeuralNetwork
from MachineLearning.Models.experiment_pure.classical_support_vector_machine import SupportVectorMachine
from MachineLearning.Models.experiment_pure.quantum_support_vector_machine import QuantumSupportVectorMachine


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

        if model_class == SupportVectorMachine:

            X_train, y_train = self._dataset_to_numpy(self.trainset)
            X_val, y_val = self._dataset_to_numpy(self.valset)

            clf = make_pipeline(
                StandardScaler(),
                SupportVectorMachine(
                    kernel=model_config.get("kernel", "rbf"),
                    C=model_config.get("C", 1.0),
                    gamma=model_config.get("gamma", "scale"),
                    degree=model_config.get("degree", 3),
                    coef0=model_config.get("coef0", 0.0),
                )
            )

            clf.fit(X_train, y_train)

            val_preds = clf.predict(X_val)
            val_metrics = self.calculate_metrics(y_val, val_preds)

            metrics_dictionary["training_data"].append({
                "accuracy": [100 * val_metrics["accuracy"]]
            })

            metrics = self.test_model(clf)
            metrics_dictionary["testing_data"].append(metrics)

            return clf, metrics_dictionary

        elif model_class == QuantumSupportVectorMachine:

            path = "../Results/Experiment_5/kernels"
            kernels = self.load_kernels(path)

            K_train = kernels[("train",model_config.get("encoding", 'angle'))]
            K_val = kernels[("validation", model_config.get("encoding", 'angle'))]
            K_test = kernels[("test",model_config.get("encoding", 'angle'))]

            _, y_train = self._dataset_to_numpy(self.trainset)
            _, y_val = self._dataset_to_numpy(self.valset)
            _, y_test = self._dataset_to_numpy(self.testset)

            clf = model_class(
                C=model_config.get("C", 1.0),
                encoding=model_config.get("encoding", 'angle')
            )

            clf.fit(K_train, y_train)

            accuracy = clf.score(K_val, y_val)
            metrics_dictionary["training_data"].append({
                "accuracy": [100 * accuracy],
            })

            preds = clf.predict(K_test)
            metrics = self.calculate_metrics(y_test, preds)
            metrics_dictionary["testing_data"].append(metrics)

            return clf, metrics_dictionary


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

                    if isinstance(net, QuantumNeuralNetwork):
                        outputs = net(inputs).squeeze()
                        probs = torch.clamp((outputs + 1.0) / 2.0, 0.0, 1.0)
                        loss = self.criterion(probs, labels.float())
                    elif isinstance(net, ClassicalNeuralNetwork):
                        outputs = net(inputs).squeeze()
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

                        elif isinstance(net, ClassicalNeuralNetwork):
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

    def load_kernels(self, path):
        directory = Path(path)
        kernels = {}
        for file in directory.iterdir():
            with open(file, "rb") as f:
                kernel_dict = pickle.load(f)
            kernels.update(kernel_dict)
        return kernels

    def _dataset_to_numpy(self, dataset):
        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        X, y = next(iter(loader))
        X = X.detach().cpu().numpy().astype(np.float64, copy=False)
        y = y.detach().cpu().numpy().astype(np.int64, copy=False)
        return X, y

    def test_model(self, model, device="cpu", num_workers=2):

        if isinstance(model, (SupportVectorMachine, Pipeline)):
            X_test, y_test = self._dataset_to_numpy(self.testset)
            preds = model.predict(X_test)
            return self.calculate_metrics(y_test, preds)

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
                elif isinstance(model, ClassicalNeuralNetwork):
                    _, preds = torch.max(outputs.data, 1)

                all_preds.append(preds)
                all_labels.append(labels)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        metrics = self.calculate_metrics(all_labels,all_preds)

        return metrics

    def calculate_metrics(self, labels, predictions):
        if torch.is_tensor(labels):
            labels = labels.detach().cpu().numpy()
        else:
            labels = np.asarray(labels)

        if torch.is_tensor(predictions):
            predictions = predictions.detach().cpu().numpy()
        else:
            predictions = np.asarray(predictions)


        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        accuracy = accuracy_score(labels, predictions)
        balanced_acc = balanced_accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, zero_division=0, average='macro')
        recall = recall_score(labels, predictions, zero_division=0, average='macro')
        f1 = f1_score(labels, predictions, zero_division=0, average='macro')

        return {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}
        }