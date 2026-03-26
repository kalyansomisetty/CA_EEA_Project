import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from Config import *
import random
seed = 0
random.seed(seed)
np.random.seed(seed)

class Data():
    """
    Encapsulation class for model input data.
    Manages X_train, X_test, y_train, y_test and maintains consistent input format.
    Removes classes with insufficient instances (< MIN_CLASS_COUNT).
    """
    def __init__(self,
                 X: np.ndarray,
                 df: pd.DataFrame,
                 label_col: str = 'y') -> None:
        """
        Initialize Data object with embeddings and dataframe.
        
        Args:
            X: Feature embeddings (numpy array)
            df: Dataframe containing labels
            label_col: Column name to use as target label (default: 'y')
        """
        X_DL = df[Config.TICKET_SUMMARY] + ' ' + df[Config.INTERACTION_CONTENT]
        X_DL = X_DL.to_numpy()
        
        # Get label from specified column
        y = df[label_col].to_numpy()
        y_series = pd.Series(y)

        # Filter classes with at least MIN_CLASS_COUNT instances
        good_y_value = y_series.value_counts()[y_series.value_counts() >= Config.MIN_CLASS_COUNT].index

        if len(good_y_value) < 1:
            print("None of the class have more than 3 records: Skipping ...")
            self.X_train = None
            return

        # Keep only good classes
        y_good = y[y_series.isin(good_y_value)]
        X_good = X[y_series.isin(good_y_value)]

        # Adjust test size based on filtered data
        new_test_size = X.shape[0] * Config.TEST_SIZE / X_good.shape[0]

        # Train-test split with stratification
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_good, y_good, 
            test_size=new_test_size, 
            random_state=Config.RANDOM_STATE, 
            stratify=y_good
        )
        self.y = y_good
        self.classes = good_y_value
        self.embeddings = X
        
    def get_type(self):
        return self.y
    
    def get_X_train(self):
        return self.X_train
    
    def get_X_test(self):
        return self.X_test
    
    def get_type_y_train(self):
        return self.y_train
    
    def get_type_y_test(self):
        return self.y_test
    
    def get_embeddings(self):
        return self.embeddings
    
    def get_classes(self):
        return self.classes
