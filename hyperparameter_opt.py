import utils.data as du
import utils.plots as pu
import utils.training as tu
import utils.models as models

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.utils.data as tdu
import numpy as np
import mlflow
import optuna
from tqdm import tqdm

def objective(trial):
    mlflow.start_run()
    scan_dataset = du.PosDataDatset('../../Data/robo_steer/raw/', 200)
    train_size = int(0.8 * len(scan_dataset))
    test_size = len(scan_dataset) - train_size
    train_dataset, test_dataset = tdu.random_split(scan_dataset, [train_size, test_size])

    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    mlflow.log_param('batch_size', batch_size)
    train_loader = tdu.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = tdu.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    use_cnn = False
    if use_cnn:
        model = models.ConvNet(1)
    else:
        hidden_size = trial.suggest_categorical('hidden_size', [256, 512, 1024])
        mlflow.log_param('hidden_size', hidden_size)
        num_layers = trial.suggest_int("num_layers", 1, 3)
        mlflow.log_param('num_layers', num_layers)
        model = models.MLP(200, hidden_size, 1, num_layers)


    lr = trial.suggest_float("lr", 1e-7, 1e-1, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["AdamW", "RMSprop", "SGD"])
    mlflow.log_param('optimizer', optimizer_name)
    mlflow.log_param('lr', lr)
    mlflow.log_param('is_cnn', use_cnn)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    #criterion = nn.MSELoss()
    criterion = nn.L1Loss()
    num_epochs = 150

    trainer = tu.Trainer(
        train_loader,
        test_loader,
        model, 
        optimizer,
        criterion
    )

    losses = []
    for epoch in tqdm(range(num_epochs)):
        trainer.train(1)
        loss = trainer.test_model_loss(test_loader, model, criterion)
        trial.report(loss, epoch)
        losses.append(loss)
        if trial.should_prune():
            mlflow.end_run()
            raise optuna.exceptions.TrialPruned()

    """
    if use_cnn:
        torch.save(model.state_dict(), f'models/cnn.ckpt')
        mlflow.log_artifact(f'models/cnn.ckpt')
    else:
        torch.save(model.state_dict(), f'models/mlp_{hidden_size}_{num_layers}.ckpt')
        mlflow.log_artifact(f'models/mlp_{hidden_size}_{num_layers}.ckpt')
    """

    mlflow.end_run()
    return np.min(losses)

if __name__ == '__main__':
    mlflow.set_experiment('HyperOpt')
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=500, catch=())