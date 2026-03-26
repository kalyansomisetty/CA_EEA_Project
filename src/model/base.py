from abc import ABC, abstractmethod
import pandas as pd
import numpy as np


class BaseModel(ABC):
    """
    Abstract base class for all machine learning models.
    Defines the interface that all model implementations must follow.
    Ensures consistent interaction with the controller.
    """
    def __init__(self) -> None:
        pass

    @abstractmethod
    def train(self, data) -> None:
        """
        Train the model using the provided Data object.
        
        Args:
            data: Data object containing X_train, y_train, etc.
        """
        pass

    @abstractmethod
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions on test data.
        
        Args:
            X_test: Test feature matrix
            
        Returns:
            Predicted labels
        """
        pass

    @abstractmethod
    def print_results(self, data) -> None:
        """
        Print classification results and metrics.
        
        Args:
            data: Data object containing test labels
        """
        pass

    def data_transform(self) -> None:
        """
        Optional data transformation specific to model.
        """
        pass
