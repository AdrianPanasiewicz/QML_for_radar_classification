from tqdm import tqdm
from sklearn.pipeline import Pipeline
import pennylane as qml
import numpy as np
import torch
import pickle
import gc
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from pathlib import Path
import optuna
from MachineLearning.Trainers.abstract_trainer import AbstractTrainer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from MachineLearning.Models.experiment_pure.quantum_neural_network import QuantumNeuralNetwork
from MachineLearning.Models.experiment_pure.classical_neural_network import ClassicalNeuralNetwork
from MachineLearning.Models.experiment_pure.classical_support_vector_machine import SupportVectorMachine
from MachineLearning.Models.experiment_pure.quantum_support_vector_machine import QuantumSupportVectorMachine
from MachineLearning.Trainers.utils import square_kernel_matrix, kernel_matrix

class HyperparameterTrainer(AbstractTrainer):
    def __init__(self, training_path, validating_path, testing_path , criterion):
        super().__init__(training_path, validating_path, testing_path, criterion)

    def train_model(self, trial, config, model_class):
        training_config = config["training_config"]
        model_config = config["model_config"]

        if model_class == SupportVectorMachine:

            X_train, y_train = self._dataset_to_numpy(self.trainset)
            X_val, y_val = self._dataset_to_numpy(self.valset)
            X_test, y_test = self._dataset_to_numpy(self.testset)

            clf = model_class(
                    kernel=model_config.get("kernel", "rbf"),
                    C=model_config.get("C", 1.0),
                    gamma=model_config.get("gamma", "scale"),
                    degree=model_config.get("degree", 3),
                    coef0=model_config.get("coef0", 0.0),
            )


            clf.fit(X_train, y_train)
            val_preds = clf.predict(X_val)
            accuracy = accuracy_score(y_val, val_preds)

            trial.report(accuracy, 0)
            if trial.should_prune():
                raise optuna.TrialPruned()

            return accuracy

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
            # val_preds = clf.predict(K_val)
            accuracy = clf.score(K_val, y_val)

            trial.report(accuracy, 0)
            if trial.should_prune():
                raise optuna.TrialPruned()

            return accuracy

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

    def precompute_kernel(self, kernel_type, encoding_name, n_qubits=None):
        X_train, _ = self._dataset_to_numpy(self.trainset)
        X_val, _ = self._dataset_to_numpy(self.valset)
        X_test, _ = self._dataset_to_numpy(self.testset)

        if n_qubits is None:
            n_qubits = X_train.shape[1]

        def angle_encoding(x):
            qml.AngleEmbedding(features=x, wires=range(n_qubits), rotation='X')

        def amplitude_embedding(x):
            qml.AmplitudeEmbedding(features=x, wires=range(n_qubits), pad_with=0.0, normalize=True)

        encodings = {
            "angle": angle_encoding,
            "amplitude": amplitude_embedding,
        }

        assert kernel_type in ("train", "validation", "test"), f"{kernel_type} not in ('train', 'validation', 'test'). Please select kernel type from one of those."
        assert encoding_name in encodings.keys(), f"{encoding_name} type does not exits. Please select encoding type from the following f{encodings.keys()}"

        enc = encodings[encoding_name]
        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev)
        def kernel_qnode(x1, x2):
            enc(x1)
            qml.adjoint(enc)(x2)
            return qml.expval(qml.Projector([0] * n_qubits, wires=range(n_qubits)))

        match kernel_type:
            case "train":
                K = square_kernel_matrix(X_train, kernel_qnode)
            case "validation":
                K = kernel_matrix(X_val, X_train, kernel_qnode)
            case "test":
                K = kernel_matrix(X_test, X_train, kernel_qnode)
            case _:
                raise NotImplementedError(f"Kernel construction for f{encoding_name} and f{kernel_type} is not implemented.")

        kernel = {(kernel_type, encoding_name) : K}

        return kernel

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

        if isinstance(model, QuantumSupportVectorMachine):
            path = "../Results/Experiment_5/kernels"
            kernels = self.load_kernels(path)

            K_test = kernels[("test",model.encoding)]
            _, y_test = self._dataset_to_numpy(self.testset)
            preds = model.predict(K_test)

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