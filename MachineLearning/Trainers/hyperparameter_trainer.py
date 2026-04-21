import tempfile
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import SGD
from ray import tune
from ray.tune import Checkpoint
from MachineLearning.Trainers.abstract_trainer import AbstractTrainer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class TrainerForHyperparameterSearch(AbstractTrainer):
    def __init__(self, training_path, validating_path, testing_path , criterion):
        super().__init__(training_path, validating_path, testing_path, criterion)

    def train_model(self, config, model_class: nn.Module):
        net = model_class(config["layers"], config["neurons_per_layer"])
        device = config["device"]

        net = net.to(device)
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)

        optimizer = SGD(net.parameters(), lr=config["lr"], momentum=0.9)

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

        trainloader = DataLoader(self.trainset, batch_size=int(config["batch_size"]), shuffle=True)
        valloader = DataLoader(self.valset, batch_size=int(config["batch_size"]))

        for epoch in range(start_epoch, config['epochs']):
            running_loss = 0.0
            epoch_steps = 0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = net(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()


                epoch_steps += 1
                # running_loss += loss.item()
                # if i % 2000 == 1999:
                #     print(
                #         "[%d, %5d] loss: %.3f"
                #         % (epoch + 1, i + 1, running_loss / epoch_steps)
                #     )
                #     running_loss = 0.0

            val_loss = 0.0
            val_steps = 0
            total = 0
            correct = 0
            for i, data in enumerate(valloader, 0):
                with torch.no_grad():
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = net(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    loss = self.criterion(outputs, labels)
                    val_loss += loss.cpu().numpy()
                    val_steps += 1

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
                tune.report(
                    {"loss": val_loss / val_steps, "accuracy": correct / total},
                    checkpoint=checkpoint,
                )


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
                val_loss += loss.item() * inputs.size(0)

                all_preds.append(preds)
                all_labels.append(labels)


        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        results = self.calculate_metrics(all_labels,all_preds)

        return results

    def calculate_metrics(self, labels, predictions):

        tn, fp, fn, tp = confusion_matrix(labels, predictions)
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