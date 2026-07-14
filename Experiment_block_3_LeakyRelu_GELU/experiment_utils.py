import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt
import copy

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

class MLPClassifierLeakyReLU(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Dropout(0.2),

            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.network(x)

class MLPClassifierGELU(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.GELU(),
            nn.Dropout(0.2),

            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.network(x)

class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 3)
        )

    def forward(self, x):
        return self.network(x)


def run_one_experiment(
    model,
    criterion,
    optimizer,
    X_train_tensor,
    y_train_tensor,
    X_test_tensor,
    y_test_tensor,
    batch_size=16,
    num_epochs=1000,
    seed=42,
    validation_fraction=0.2,
    plot_graph=True,
    verbose=True
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_samples = X_train_tensor.shape[0]
    split_idx = int((1 - validation_fraction) * num_samples)

    X_train_inner = X_train_tensor[:split_idx]
    y_train_inner = y_train_tensor[:split_idx]

    X_val = X_train_tensor[split_idx:]
    y_val = y_train_tensor[split_idx:]

    train_dataset = TensorDataset(
        X_train_inner,
        y_train_inner
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator
    )

    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_loss = float("inf")
    best_epoch = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()

        epoch_train_loss_sum = 0.0
        epoch_train_samples = 0

        for X_batch, y_batch in train_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_batch_size = X_batch.shape[0]

            epoch_train_loss_sum += loss.item() * current_batch_size
            epoch_train_samples += current_batch_size

        avg_train_loss = epoch_train_loss_sum / epoch_train_samples
        train_losses.append(avg_train_loss)

        model.eval()

        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

            val_pred = torch.argmax(val_outputs, dim=1)

            val_accuracy = accuracy_score(
                y_val.numpy(),
                val_pred.numpy()
            )

        val_loss_value = val_loss.item()

        val_losses.append(val_loss_value)
        val_accuracies.append(val_accuracy)

        if val_loss_value < best_val_loss:
            best_val_loss = val_loss_value
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())

        if verbose and (epoch + 1) % 50 == 0:
            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"train loss = {avg_train_loss:.4f} | "
                f"val loss = {val_loss_value:.4f} | "
                f"val acc = {val_accuracy:.4f}"
            )

    model.load_state_dict(best_model_state)

    model.eval()

    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        y_pred = torch.argmax(test_outputs, dim=1)

    y_pred_np = y_pred.numpy()
    y_test_np = y_test_tensor.numpy()

    test_accuracy = accuracy_score(
        y_test_np,
        y_pred_np
    )

    if verbose:
        print()
        print("=========================")
        print(f"Best epoch:    {best_epoch}")
        print(f"Best val loss: {best_val_loss:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        print("=========================")

    if plot_graph:
        plt.figure(figsize=(8, 5))

        plt.plot(
            range(1, num_epochs + 1),
            train_losses,
            label="Train loss"
        )

        plt.plot(
            range(1, num_epochs + 1),
            val_losses,
            label="Validation loss"
        )

        plt.axvline(
            best_epoch,
            linestyle="--",
            label=f"Best epoch = {best_epoch}"
        )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Train and validation loss")
        plt.legend()
        plt.grid(True)
        plt.show()

    return test_accuracy

def calculate_specificity(y_true, y_pred):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2]
    )

    specificities = []

    for class_id in [0, 1, 2]:
        tp = cm[class_id, class_id]
        fp = cm[:, class_id].sum() - tp
        fn = cm[class_id, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        if tn + fp == 0:
            specificity = 0
        else:
            specificity = tn / (tn + fp)

        specificities.append(specificity)

    return np.mean(specificities)


def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0
    )

    specificity = calculate_specificity(
        y_true,
        y_pred
    )

    f1 = f1_score(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1
    }

def run(
    model_class,
    X_train_tensor,
    y_train_tensor,
    X_test_tensor,
    y_test_tensor,
    num_runs=10,
    num_epochs=300,
    batch_size=16,
    learning_rate=0.001
):
    all_results = []

    class_names = ["Away Win", "Draw", "Home Win"]

    input_dim = X_train_tensor.shape[1]

    last_model = None
    last_y_pred_np = None

    y_test_np = y_test_tensor.numpy()

    for run_id in range(num_runs):
        print(f"Run {run_id + 1}/{num_runs}")

        seed = 42 + run_id

        torch.manual_seed(seed)
        np.random.seed(seed)

        model = model_class(input_dim)

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate
        )

        run_one_experiment(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            X_train_tensor=X_train_tensor,
            y_train_tensor=y_train_tensor,
            X_test_tensor=X_test_tensor,
            y_test_tensor=y_test_tensor,
            batch_size=batch_size,
            num_epochs=num_epochs,
            seed=seed,
            plot_graph=False,
            verbose=False
        )

        model.eval()

        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            y_pred = torch.argmax(test_outputs, dim=1)

        y_pred_np = y_pred.numpy()

        metrics = calculate_metrics(
            y_true=y_test_np,
            y_pred=y_pred_np
        )

        metrics["run"] = run_id + 1
        metrics["seed"] = seed

        all_results.append(metrics)

        last_model = model
        last_y_pred_np = y_pred_np

        print(
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"Specificity: {metrics['specificity']:.4f} | "
            f"F1: {metrics['f1']:.4f}"
        )

        print()

    results_df = pd.DataFrame(all_results)

    summary = {
        "accuracy": results_df["accuracy"].mean(),
        "precision": results_df["precision"].mean(),
        "recall": results_df["recall"].mean(),
        "specificity": results_df["specificity"].mean(),
        "f1": results_df["f1"].mean(),
        "f1_std": results_df["f1"].std(),
        "f1_min": results_df["f1"].min(),
        "f1_max": results_df["f1"].max()
    }

    print("=========================")
    print("Summary over all runs")
    print("=========================")
    print(f"Accuracy:    {summary['accuracy']:.4f}")
    print(f"Precision:   {summary['precision']:.4f}")
    print(f"Recall:      {summary['recall']:.4f}")
    print(f"Specificity: {summary['specificity']:.4f}")
    print(f"F1 mean:     {summary['f1']:.4f}")
    print(f"F1 std:      {summary['f1_std']:.4f}")
    print(f"F1 min:      {summary['f1_min']:.4f}")
    print(f"F1 max:      {summary['f1_max']:.4f}")
    print("=========================")

    print()
    print("Classification report for last run:")

    print(
        classification_report(
            y_test_np,
            last_y_pred_np,
            target_names=class_names,
            zero_division=0
        )
    )

    cm = confusion_matrix(
        y_test_np,
        last_y_pred_np,
        labels=[0, 1, 2]
    )

    cm_df = pd.DataFrame(
        cm,
        index=[f"True {name}" for name in class_names],
        columns=[f"Pred {name}" for name in class_names]
    )

    print("Confusion matrix for last run:")
    print(cm_df)

    return summary, results_df

def save_result(
    experiment_name,
    summary,
    results_file="../results/experiment_summary_final.csv"
):
    results_file = Path(results_file)
    results_file.parent.mkdir(exist_ok=True)

    row = pd.DataFrame([{
        "experiment": experiment_name,
        "accuracy": summary["accuracy"],
        "precision": summary["precision"],
        "recall": summary["recall"],
        "specificity": summary["specificity"],
        "f1": summary["f1"],
        "f1_std": summary["f1_std"],
        "f1_min": summary["f1_min"],
        "f1_max": summary["f1_max"]
    }])

    if results_file.exists():
        old_results = pd.read_csv(results_file)

        old_results = old_results[
            old_results["experiment"] != experiment_name
        ]

        new_results = pd.concat(
            [old_results, row],
            ignore_index=True
        )
    else:
        new_results = row

    new_results.to_csv(
        results_file,
        index=False
    )

    print()
    print(f"Saved result for {experiment_name}")
    print(f"File: {results_file}")

def save_result_v1(
    experiment_name,
    summary,
    results_file="../results/experiment_summary_v1.csv"
):
    results_file = Path(results_file)
    results_file.parent.mkdir(exist_ok=True)

    row = pd.DataFrame([{
        "experiment": experiment_name,
        "accuracy": summary["accuracy"],
        "precision": summary["precision"],
        "recall": summary["recall"],
        "specificity": summary["specificity"],
        "f1": summary["f1"],
        "f1_std": summary["f1_std"],
        "f1_min": summary["f1_min"],
        "f1_max": summary["f1_max"]
    }])

    if results_file.exists():
        old_results = pd.read_csv(results_file)

        old_results = old_results[
            old_results["experiment"] != experiment_name
        ]

        new_results = pd.concat(
            [old_results, row],
            ignore_index=True
        )
    else:
        new_results = row

    new_results.to_csv(
        results_file,
        index=False
    )

    print()
    print(f"Saved result for {experiment_name}")
    print(f"File: {results_file}")

def save_result_activation_change(
    experiment_name,
    summary,
    results_file="../results/experiment_summary_activation_change.csv"
):
    results_file = Path(results_file)
    results_file.parent.mkdir(exist_ok=True)

    row = pd.DataFrame([{
        "experiment": experiment_name,
        "accuracy": summary["accuracy"],
        "precision": summary["precision"],
        "recall": summary["recall"],
        "specificity": summary["specificity"],
        "f1": summary["f1"],
        "f1_std": summary["f1_std"],
        "f1_min": summary["f1_min"],
        "f1_max": summary["f1_max"]
    }])

    if results_file.exists():
        old_results = pd.read_csv(results_file)

        old_results = old_results[
            old_results["experiment"] != experiment_name
        ]

        new_results = pd.concat(
            [old_results, row],
            ignore_index=True
        )
    else:
        new_results = row

    new_results.to_csv(
        results_file,
        index=False
    )

    print()
    print(f"Saved result for {experiment_name}")
    print(f"File: {results_file}")