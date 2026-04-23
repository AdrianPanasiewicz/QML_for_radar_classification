import tempfile
from pathlib import Path
import torch
import os
import gc
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, LBFGS
from ray import tune
from ray.tune import Checkpoint
from MachineLearning.Trainers.abstract_trainer import AbstractTrainer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from MachineLearning.Models.experiment_pure.quantum_neural_network import QuantumNeuralNetwork


class TrainerForHyperparameterSearch(AbstractTrainer):
    def __init__(self, training_path, validating_path, testing_path , criterion):
        super().__init__(training_path, validating_path, testing_path, criterion)

    def train_model(self, config, model_class):
        training_config = config["training_config"]
        model_config = config["model_config"]

        net = model_class(model_config)
        device = training_config["device"]
        net = net.to(device)

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
            raise TypeError(f"Assigned {training_config["optimizer"]["name"]} is not implemented in hyperparameter search")

        checkpoint = tune.get_checkpoint()
        if checkpoint:
            with checkpoint.as_directory() as checkpoint_dir:
                checkpoint_path = Path(checkpoint_dir) / "checkpoint.pt"
                checkpoint_state = torch.load(checkpoint_path)
                start_epoch = checkpoint_state["epoch"]
                net.load_state_dict(checkpoint_state["net_state_dict"])
                optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
        else:
            start_epoch = 0

        trainloader = DataLoader(
            self.trainset,
            batch_size=int(training_config["batch_size"]),
            shuffle=True,
            num_workers=4,
            pin_memory=False if training_config["device"]=="cpu" else True,
            persistent_workers=True
        )
        valloader = DataLoader(
            self.valset,
            batch_size=int(training_config["batch_size"]),
            shuffle = False,
            num_workers = 2,
            pin_memory = False if training_config["device"]=="cpu" else True,
            persistent_workers = True
        )

        threads = 8
        torch.set_num_threads(threads)
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)

        for epoch in range(start_epoch, training_config['epochs']):
            net.train()
            epoch_steps = 0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
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

                epoch_steps += 1


            net.eval()
            val_loss = 0.0
            val_steps = 0
            total = 0
            correct = 0
            for i, data in enumerate(valloader, 0):
                with torch.no_grad():
                    inputs, labels = data
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

            if epoch % 20 == 0:
                checkpoint_data = {
                    "epoch": epoch,
                    "model_class_name": net.module.model_name if isinstance(net, nn.DataParallel) else net.model_name,
                    "model_init_kwargs": net.module.init_kwargs if isinstance(net, nn.DataParallel) else net.init_kwargs,
                    "net_state_dict": net.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }
                with tempfile.TemporaryDirectory() as checkpoint_dir:
                    checkpoint_path = Path(checkpoint_dir) / "checkpoint.pt"
                    torch.save(checkpoint_data, checkpoint_path)

                    checkpoint = Checkpoint.from_directory(checkpoint_dir)

                    final_loss = float(val_loss.cpu().numpy()) / val_steps
                    final_accuracy = float(correct.cpu().numpy()) / total

                    tune.report(
                        {"loss": final_loss, "accuracy": final_accuracy},
                        checkpoint=checkpoint,
                    )

            try:
                del inputs, labels, outputs, loss, probs
            except NameError:
                pass
            gc.collect()


    def test_model(self, model, device="cpu"):
        testloader = DataLoader(self.testset, batch_size=32, shuffle=True)

        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)

                loss = self.criterion(outputs, labels)
                val_loss += loss.detach() * inputs.size(0)

                all_preds.append(preds)
                all_labels.append(labels)


        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        results = self.calculate_metrics(all_labels,all_preds)

        return results

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