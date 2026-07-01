import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class MLPClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )

    def forward(self, x):
        return self.network(x)


def run_experiment(
    X_train_tensor,
    y_train_tensor,
    X_test_tensor,
    y_test_tensor,
    num_runs=10,
    num_epochs=100,
    batch_size=16,
    learning_rate=0.001
):
    accuracies = []
    class_names = ["Away Win", "Draw", "Home Win"]
    input_dim = X_train_tensor.shape[1]

    for run in range(num_runs):
        print(f"Run {run + 1}/{num_runs}")

        seed = 42 + run
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )

        model = MLPClassifier(input_dim=input_dim)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate
        )

        for epoch in range(num_epochs):
            model.train()

            for X_batch, y_batch in train_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()

        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            y_pred = torch.argmax(test_outputs, dim=1)

        y_pred_np = y_pred.numpy()
        y_test_np = y_test_tensor.numpy()

        acc = accuracy_score(y_test_np, y_pred_np)
        accuracies.append(acc)

        print(f"Accuracy: {acc:.4f}")
        print()

    mean_accuracy = np.mean(accuracies)

    print("=================================")
    print(f"Mean accuracy: {mean_accuracy:.4f}")
    print("=================================")

    print("Classification report for last run:")
    print(classification_report(
        y_test_np,
        y_pred_np,
        target_names=class_names
    ))

    cm = confusion_matrix(y_test_np, y_pred_np)

    cm_df = pd.DataFrame(
        cm,
        index=[f"True {name}" for name in class_names],
        columns=[f"Pred {name}" for name in class_names]
    )

    print("Confusion matrix for last run:")
    print(cm_df)

    
