from typing import Tuple
import numpy as np
from src.logger import Logger


class DataLoader():
    """ Batch Data Loader that shuffles the data at every epoch

    Args:
        data (np.ndarray): 2D Data array
        labels (np.ndarray): 1D class/label array
        batch_size (int): Batch size
    """

    def __init__(self, data: np.ndarray, labels: np.ndarray, batch_size: int):
        self.batch_size = batch_size
        self.data = self.preprocess(data)
        self.labels = labels
        self.index = None

    def __iter__(self) -> "DataLoader":
        """ Shuffle the data and reset the index

        Returns:
            DataLoader: self object
        """
        self.shuffle()
        self.index = 0
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Return a batch of data and label starting at <self.index>. Also increment <self.index>.

        Raises:
            StopIteration: If builtin "next" function is called when the data is fully passed 

        Returns:
            Tuple[np.ndarray, np.ndarray]: Batch of data (B, D) and label (B) arrays
        """

        if self.index >= len(self.data):
            raise StopIteration
        
        batch_data = self.data[self.index:self.index + self.batch_size]
        batch_label = self.labels[self.index:self.index + self.batch_size]
        
        self.index += self.batch_size
        

        return batch_data, batch_label

    def reset(self):
        """ Reset the index for the next iteration. """
        self.index = 0


    def shuffle(self) -> None:
        """Shuffle the data and labels in unison."""
        assert len(self.data) == len(self.labels), "Data and labels must be of the same length"
        
        permutation = np.random.permutation(len(self.data))
        self.data = self.data[permutation]
        self.labels = self.labels[permutation]


    @staticmethod
    def preprocess(data: np.ndarray) -> np.array:
        """ Preprocess the data

        Args:
            data (np.ndarray): data array

        Returns:
            np.array: Float data array with values ranging from 0 to 1 
        """
        return data.astype(np.float32) / 255


class LogisticRegresssionClassifier():
    """ Logistic Regression Classifier

    Args:
        n_features (int): Number of features
        n_classes (int): Number of unique classes/labels
    """

    def __init__(self, n_features: int, n_classes: int):
        self.n_features = n_features
        self.n_classes = n_classes
        self.weights, self.bias = self._initialize()

    def _initialize(self) -> Tuple[np.ndarray, np.ndarray]:
        """ Initialize weights and biases

        Returns:
            Tuple[np.ndarray, np.ndarray]: weight (D, C) and bias (C) arrays
        """
        
        self.weights = 5 * np.random.randn(self.n_features, self.n_classes)
        
        self.bias = 4 * np.random.randn(self.n_classes)


        return self.weights, self.bias


    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """ Return class probabilities of the given input array 

        Args:
            inputs (np.ndarray): input array of shape (B, D)

        Returns:
            np.ndarray: Class probabilities of shape (B, C)

        """
        
        prediction = inputs  @ self.weights + self.bias


        
        prediction = self.softmax(prediction)

        return prediction


    def fit(self,
            train_data_loader: DataLoader,
            eval_data_loader: DataLoader,
            learning_rate: float,
            l2_regularization_coeff: float,
            epochs: int,
            logger: Logger) -> None:
        """ Main training function

        Args:
            train_data_loader (DataLoader): Data loader with training data
            eval_data_loader (DataLoader): Data loader with evaluation data
            learning_rate (float): Learning rate
            l2_regularization_coeff (float): L2 regularization coefficient
            epochs (int): Number of epochs
            logger (Logger): Logger object for logging accuracies and losses
        """

        for epoch_index in range(epochs):
            train_epoch_accuracy_list = []
            train_epoch_loss_list = []
            eval_epoch_accuracy_list = []

            for iter_index, (train_data, train_label) in enumerate(train_data_loader):

                probs = self.predict(train_data)
                predictions = probs.argmax(axis=-1)
                train_loss = self.nll_loss(probs, train_label)
                train_accuracy = self.accuracy(predictions, train_label)

                gradient = self.nll_gradients(probs, train_data, train_label)
                self.update(gradient, learning_rate, l2_regularization_coeff)

                train_epoch_accuracy_list.append(train_accuracy)
                train_epoch_loss_list.append(train_loss)
                logger.iter_train_accuracy.append(train_accuracy)
                logger.iter_train_loss.append(train_loss)
                logger.log_iteration(epoch_index, iter_index)

            for eval_data, eval_lavel in eval_data_loader:
                probs = self.predict(eval_data)
                predictions = probs.argmax(axis=-1)
                eval_accuracy = self.accuracy(predictions, eval_lavel)
                eval_epoch_accuracy_list.append(eval_accuracy)

            logger.epoch_train_accuracy.append(np.mean(train_epoch_accuracy_list))
            logger.epoch_eval_accuracy.append(np.mean(eval_epoch_accuracy_list))
            logger.log_epoch(epoch_index)

    def update(self, gradients: Tuple[np.ndarray, np.ndarray], learning_rate: float, l2_regularization_coeff: float) -> None:
        """ Update weight and biases with the given gradients and regularization 

        Args:
            gradients (Tuple[np.ndarray, np.ndarray]): gradient of weights and biases [(D, C), (C)]
            learning_rate (float): Learning rate
            l2_regularization_coeff (float): L2 regularization coefficient
        """
        self.weights -= learning_rate* (gradients[0] + l2_regularization_coeff*self.weights)
        self.bias -= (learning_rate*gradients[1])
        


    @staticmethod
    def nll_gradients(probs: np.ndarray, inputs: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """ Calculate the gradients of negative log likelihood loss with respect to weights and biases

        Args:
            probs (np.ndarray): Softmax output of shape: (B, C)
            inputs (np.ndarray): Input array of shape: (B, D)
            labels (np.ndarray): True class labels: (B,)

        Returns:
            Tuple[np.ndarray, np.ndarray]: Gradients with respect to weights and biases. Shape: [(D, C), (C,)]
        """
        
        """
        B , C = probs.shape
        D = inputs.shape[1] 

        weight_grad = np.zeros((D, C))  
        bias_grad = np.zeros(C)         

        # Calculate gradients
        
        for i in range(B):
            difference = labels[i] - probs[i]
            transposed_input = inputs[i].T
            
            if difference == 0:
                
                grad_i = probs[i]* (1-probs[i])
                
                weight_grad[i] = (1/difference) * grad_i * transposed_input
                
                bias_grad[i] = (1/difference) * grad_i
            else : 
                weight_grad[i] = 0
                bias_grad[i] = 0
                
                
                
                
        """
        
        n_classes= probs.shape[-1]
        
        
        one_hot_label = (np.arange(n_classes).reshape(1,-1) == labels.reshape(-1,1) ).astype(np.float32)
        
        
        weight_grad =-inputs.T @ (one_hot_label - probs)
        bias_grad = -(one_hot_label - probs)

        return weight_grad, bias_grad.sum(0) 

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        """ Softmax function

        Args:
            logits (np.ndarray): input array of shape (B, C)

        Returns:
            np.ndarray: output array of shape (B, C)
        """
        
        softmax_val = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return softmax_val / np.sum(softmax_val, axis=1, keepdims=True)

    @staticmethod
    def accuracy(prediction: np.ndarray, label: np.ndarray) -> np.float32:
        """ Calculate mean accuracy

        Args:
            prediction (np.ndarray): Prediction array of shape (B)
            label (np.ndarray): Ground truth array of shape (B)

        Returns:
            np.float32: Average accuracy
        """
        
        correct = np.sum(prediction == label)  # Count number of correct predictions
     
        return np.float32(correct / len(label))


    @staticmethod
    def nll_loss(prediction_probs: np.ndarray, label: np.ndarray) -> np.float32:
        """ Calculate mean negative log likelihood

        Args:
            prediction_probs (np.ndarray): Prediction probabilities of shape (B, C)
            label (np.ndarray): Ground truth array of shape (B)

        Returns:
            np.float32: Mean negative log likelihood loss
        """
        
      
        B = prediction_probs.shape[0]
        loss = 0.0
        epsilon = 1e-15  # Small value to avoid log(0)
        
        #loglikelihoods = (np.log(prediction_probs)[np.arange(label.shape[0]), label]).mean()
        
        for i in range(B):
            predicted_probs_i = prediction_probs[i]
            true_label_i = label[i]
            nll_i = -np.log(predicted_probs_i[true_label_i] + epsilon)
            loss += nll_i
        
        mean_nll_loss = loss / B  # Corrected to calculate mean properly
        
        return np.float32(mean_nll_loss)

    

    @staticmethod
    def confusion_matrix(label: np.ndarray, predictions: np.ndarray, n_classes: np.ndarray) -> np.ndarray:
        """ Calculate confusion matrix

        Args:
            predictions (np.ndarray): Prediction array of shape (B)
            label (np.ndarray): Ground truth array of shape (B)

        Returns:
            np.ndarray: Confusion matrix of shape (C, C)
        """
        conf_matrix = np.zeros((n_classes, n_classes))
        
        for  i in range(len(label)):
            true_label = label[i]
            pred_label = predictions[i]
            
            conf_matrix[true_label, pred_label] +=1
             
        return conf_matrix





