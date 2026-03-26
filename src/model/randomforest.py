import numpy as np
import pandas as pd
from model.base import BaseModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import random

seed = 0
np.random.seed(seed)
random.seed(seed)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 200)


class RandomForest(BaseModel):
    """
    Random Forest Classifier implementation inheriting from BaseModel.
    """
    def __init__(self,
                 model_name: str,
                 embeddings: np.ndarray,
                 y: np.ndarray) -> None:
        """
        Initialize Random Forest model.
        
        Args:
            model_name: Name identifier for the model
            embeddings: Feature embeddings
            y: Target labels
        """
        super(RandomForest, self).__init__()
        self.model_name = model_name
        self.embeddings = embeddings
        self.y = y
        self.mdl = RandomForestClassifier(
            n_estimators=1000, 
            random_state=seed, 
            class_weight='balanced_subsample'
        )
        self.predictions = None
        self.data_transform()

    def train(self, data) -> None:
        """
        Train the Random Forest model.
        
        Args:
            data: Data object containing X_train and y_train
        """
        self.mdl = self.mdl.fit(data.get_X_train(), data.get_type_y_train())

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions on test data.
        
        Args:
            X_test: Test feature matrix
            
        Returns:
            Predicted labels
        """
        predictions = self.mdl.predict(X_test)
        self.predictions = predictions
        return predictions

    def print_results(self, data) -> None:
        """
        Print classification report and confusion matrix.
        
        Args:
            data: Data object containing test labels
        """
        print(f"Model: {self.model_name}")
        print(classification_report(data.get_type_y_test(), self.predictions, zero_division=0))
        print(f"Accuracy: {accuracy_score(data.get_type_y_test(), self.predictions):.4f}")

    def data_transform(self) -> None:
        """
        No specific data transformation needed for Random Forest.
        """
        pass
