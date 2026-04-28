import tempfile
from pathlib import Path
from tqdm import tqdm
import torch
import os
import gc
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, LBFGS
from ray import tune
import optuna
from ray.tune import Checkpoint
from MachineLearning.Trainers.abstract_trainer import AbstractTrainer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from MachineLearning.Models.experiment_pure.quantum_neural_network import QuantumNeuralNetwork

class HyperparameterTrainer(AbstractTrainer):
    def __init__(self, training_path, validating_path, testing_path , criterion):
        super().__init__(training_path, validating_path, testing_path, criterion)

    def train_model(self, trial, config, model_class):
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

        # checkpoint = tune.get_checkpoint()
        # if checkpoint:
        #     with checkpoint.as_directory() as checkpoint_dir:
        #         checkpoint_path = Path(checkpoint_dir) / "checkpoint.pt"
        #         checkpoint_state = torch.load(checkpoint_path)
        #         start_epoch = checkpoint_state["epoch"]
        #         net.load_state_dict(checkpoint_state["net_state_dict"])
        #         optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
        # else:
        #     start_epoch = 0

        trainloader = DataLoader(
            self.trainset,
            batch_size=int(training_config["batch_size"]),
            shuffle=True,
            num_workers=training_config['number_of_training_workers'],
            pin_memory=False if training_config["device"]=="cpu" else True,
        )
        valloader = DataLoader(
            self.valset,
            batch_size=int(training_config["batch_size"]),
            shuffle = False,
            num_workers = training_config['number_of_validating_workers'],
            pin_memory = False if training_config["device"]=="cpu" else True,
        )

        for epoch in tqdm(range(training_config['epochs']), desc="Epochs"):
            net.train()
            epoch_steps = 0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = net(inputs).squeeze()
                if isinstance(net, QuantumNeuralNetwork):
                    probs = torch.clamp((outputs + 1.0) / 2.0, 0.0, 1.0)
                    loss = self.criterion(probs, labels.float())
                else:
                    loss = self.criterion(outputs, labels.long())

                if training_config["regularization"]["type"] is not None:
                    if training_config["regularization"]['type']=='l1':
                        penality = sum(p.abs().sum() for p in net.parameters())
                        loss = loss + training_config["regularization"]["lambda"] * penality

                    elif training_config['regularization']['type']=='l2':
                        penality = sum((p**2).sum() for p in net.parameters())
                        loss = loss + training_config["regularization"]["lambda"] * penality

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

            # if epoch % 20 == 0:
            #     checkpoint_data = {
            #         "epoch": epoch,
            #         "model_class_name": net.module.model_name if isinstance(net, nn.DataParallel) else net.model_name,
            #         "model_init_kwargs": net.module.init_kwargs if isinstance(net, nn.DataParallel) else net.init_kwargs,
            #         "net_state_dict": net.state_dict(),
            #         "optimizer_state_dict": optimizer.state_dict(),
            #     }
            #     with tempfile.TemporaryDirectory() as checkpoint_dir:
            #         checkpoint_path = Path(checkpoint_dir) / "checkpoint.pt"
            #         torch.save(checkpoint_data, checkpoint_path)
            #
            #         checkpoint = Checkpoint.from_directory(checkpoint_dir)
            #
            #         final_loss = float(val_loss.cpu().numpy()) / val_steps
            #         final_accuracy = float(correct.cpu().numpy()) / total
            #
            #         tune.report(
            #             {"loss": final_loss, "accuracy": final_accuracy},
            #             checkpoint=checkpoint,
            #         )

            accuracy = correct / total
            trial.report(accuracy, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

            try:
                del inputs, labels, outputs, loss, probs
            except NameError:
                pass

            gc.collect()

        del trainloader
        del valloader
        
        return accuracy


    def test_model(self, net, device="cpu"):
        testloader = DataLoader(self.testset, batch_size=32, shuffle=True)

        net.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                if isinstance(net, QuantumNeuralNetwork):
                    preds = 0 if outputs < 0 else 1
                else:
                    _, preds = torch.max(outputs.data, 1)

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