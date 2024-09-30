from typing import Tuple
import requests
import gzip
import numpy as np
import os
from dataclasses import dataclass

from src.logistic_regression import DataLoader, LogisticRegresssionClassifier
from src.logger import Logger


train_labels_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz"
train_images_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz"
test_labels_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz"
test_images_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz"

os.makedirs("data", exist_ok=True)


def download_and_load(url: str, name: str, kind: str, offset: int) -> np.ndarray:
    file_path = f"data/{kind}_{name}"
    if not os.path.exists(file_path):
        response = requests.get(url)
        with open(file_path, "wb") as f:
            f.write(response.content)
    with gzip.open(file_path, "rb") as lbpath:
        return np.frombuffer(lbpath.read(), dtype=np.uint8, offset=offset)


train_labels = download_and_load(train_labels_url, "train", "labels", 8)
train_data = download_and_load(train_images_url, "train", "images",
                               16).reshape(len(train_labels), 784)
test_labels = download_and_load(test_labels_url, "test", "labels", 8)
test_data = download_and_load(test_images_url, "test", "images", 16).reshape(len(test_labels), 784)

# %%

# train_loader = DataLoader(train_data, train_labels, batch_size=32)
# iterator = iter(train_loader)
# batch_data, batch_label = next(iterator)

# classifier = LogisticRegresssionClassifier(784, 10)
# classifier.predict(next(iterator)[0])
# label = np.array(1, dtype=np.int32)

# inccorect_pred = np.zeros((1, 10), dtype=np.float32)
# inccorect_pred[0, 0] = 1.0
#
# correct_pred = np.zeros((1, 10), dtype=np.float32)
# correct_pred[0, 1] = 1.0
#
# inputs = np.ones((1, 784), dtype=np.float32)
#
# weight_grad, bias_grad = classifier.nll_gradients(probs=inccorect_pred, inputs=inputs, labels=label)
#
# prev_weights = classifier.weights.copy()
# prev_bias = classifier.bias.copy()
# classifier.update((np.ones_like(prev_weights), np.ones_like(prev_bias)), 1.0, 0.0)

#%%

def split_dataset(data: np.ndarray, label: np.ndarray, batch_size: int, train_ratio: float = 0.9
                  ) -> Tuple[DataLoader, DataLoader]:
    """ Split the data into train and eval sets

    Args:
        data (np.ndarray): Data array of shape (B, D)
        label (np.ndarray): Label array os shape (B)
        batch_size (int): Batch size of the dataloaders
        train_ratio (float): Ratio of the train sample size to overall sample size

    Returns:
        Tuple[DataLoader, DataLoader]: Train and Eval Dataloaders
    """
    z = int(data.shape[0] * train_ratio)

    train_data, eval_data = data[:z, :], data[z:, :] # (5400,784) , (600,784)
    train_labels, eval_labels = label[:z], label[z:] # (5400) , (600)

    train_loader = DataLoader(train_data, train_labels, batch_size=batch_size)
    eval_loader = DataLoader(eval_data, eval_labels, batch_size=batch_size)

    return train_loader, eval_loader


@dataclass
class Hyperparameters():
    train_eval_ratio: float = 0.9
    batch_size: int = 32
    learning_rate: float = 1e-3
    l2_coeff: float = 1e-2
    epoch: int = 5


hyperparams = Hyperparameters()

train_loader, eval_loader = split_dataset(train_data, train_labels, hyperparams.batch_size)
logger = Logger(smooth_window_len=100, verbose=False, live_figure_update=True)
logger.render()


model = LogisticRegresssionClassifier(784, 10)
model.fit(train_loader,
          eval_loader,
          hyperparams.learning_rate,
          hyperparams.l2_coeff,
          hyperparams.epoch,
          logger)


def test_classifier(data_loader: DataLoader, model: LogisticRegresssionClassifier) -> np.ndarray:
    """ Run the model with test data loader and return confusion matrix

    Args:
        data_loader (DataLoader): Data loader of the test data
        model (LogisticRegresssionClassifier): Trained classifier

    Returns:
        np.ndarray: Confusion matrix
    """

    cs = np.zeros((model.n_classes, model.n_classes))
    for iter_index, (data, labels) in enumerate(data_loader):
        probs = model.predict(data)
        predictions = probs.argmax(axis=-1)
        cs += model.confusion_matrix(labels, predictions)
    return cs


test_loader = DataLoader(test_data, test_labels, hyperparams.batch_size)
confusion_matrix = test_classifier(test_loader, model)