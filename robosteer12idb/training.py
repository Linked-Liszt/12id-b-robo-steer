import mlflow
import torch

class Trainer():
    """
    Generic training harness for a pytorch
    Consecutive calls to train will automatically log
    results to mlflow and keep track of latest steps/epochs.
    """
    def __init__(self, train_loader, test_loader, model, optimizer, criterion):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.mlflow_step = 0
    
    def train(self, test_interval):
        for batch_num, data in enumerate(self.train_loader):
            if batch_num % test_interval == 0:
                test_loss = self.test_model_loss(self.test_loader, self.model, self.criterion)
                mlflow.log_metric('test_loss', test_loss, self.mlflow_step)

            idxs, reading, labels = data

            self.optimizer.zero_grad()
            outputs = self.model(reading)
            loss = self.criterion(outputs, labels)
            mlflow.log_metric('loss', loss.item(), self.mlflow_step)
            loss.backward()
            self.optimizer.step()
            self.mlflow_step += 1

        
    
    def test_model_loss(self, test_loader, model, criterion):
        total_loss = 0.0 
        with torch.no_grad():
            for data in test_loader:
                _, reading, labels = data
                outputs = model(reading)
                total_loss += criterion(outputs, labels).item()

        return total_loss / len(test_loader) 