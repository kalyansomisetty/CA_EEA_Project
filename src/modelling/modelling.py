from model.randomforest import RandomForest
from modelling.data_model import Data
import pandas as pd
import numpy as np
from Config import *
import warnings
warnings.filterwarnings('ignore')


def model_predict_single(data: Data, model_type: str = "RandomForest") -> None:
    """
    Train and predict using a single model on a single label.
    
    Args:
        data: Data object containing train/test data
        model_type: Type of model to use (default: RandomForest)
    """
    if model_type == "RandomForest":
        model = RandomForest("RandomForest", data.get_embeddings(), data.get_type())
        model.train(data)
        model.predict(data.get_X_test())
        model.print_results(data)


def chained_multi_output_classification(df: pd.DataFrame, X: np.ndarray) -> dict:
    """
    Chained Multi-Output Classification Strategy.
    
    A single model instance predicts progressively chained labels:
    - Level 1: Type2
    - Level 2: Type2 + Type3 (combined label)
    - Level 3: Type2 + Type3 + Type4 (combined label)
    
    Args:
        df: Dataframe with y2, y3, y4 columns
        X: Feature embeddings
        
    Returns:
        Dictionary with results for each level
    """
    results = {}
    
    # Level 1: Type2 prediction
    print("\nLevel 1 - Type2 Classification")
    print("-" * 50)
    df_l1 = df.copy()
    df_l1['y'] = df_l1['y2']
    data = Data(X, df_l1, label_col='y')
    if data.X_train is not None:
        model_predict_single(data)
        results['Level1'] = {'label': 'Type2'}
    
    # Level 2: Type2 + Type3 (combined)
    print("\nLevel 2 - Type2 + Type3 Classification")
    print("-" * 50)
    df_l2 = df.copy()
    df_l2['y'] = df_l2['y2'].astype(str) + "_" + df_l2['y3'].astype(str)
    data = Data(X, df_l2, label_col='y')
    if data.X_train is not None:
        model_predict_single(data)
        results['Level2'] = {'label': 'Type2 + Type3'}
    
    # Level 3: Type2 + Type3 + Type4 (combined)
    print("\nLevel 3 - Type2 + Type3 + Type4 Classification")
    print("-" * 50)
    df_l3 = df.copy()
    df_l3['y'] = df_l3['y2'].astype(str) + "_" + df_l3['y3'].astype(str) + "_" + df_l3['y4'].astype(str)
    data = Data(X, df_l3, label_col='y')
    if data.X_train is not None:
        model_predict_single(data)
        results['Level3'] = {'label': 'Type2 + Type3 + Type4'}
    
    return results


def hierarchical_modelling(df: pd.DataFrame, X: np.ndarray) -> dict:
    """
    Hierarchical Modelling Strategy.
    
    Multiple models chained with dataset filtering by predicted labels:
    - Model 1: Predict Type2
    - Model 2: For each Type2 class, predict Type3
    - Model 3: For each Type3 class, predict Type4
    
    Args:
        df: Dataframe with y2, y3, y4 columns
        X: Feature embeddings
        
    Returns:
        Dictionary with results organized by hierarchy
    """
    results = {}
    
    # Level 1: Type2 prediction
    print("\nLevel 1 - Type2 Hierarchical Classification")
    print("-" * 50)
    df_level1 = df.copy()
    df_level1['y'] = df_level1['y2']
    data_level1 = Data(X, df_level1, label_col='y')
    
    if data_level1.X_train is None:
        return results
    
    model_level1 = RandomForest("RandomForest_Type2", data_level1.get_embeddings(), data_level1.get_type())
    model_level1.train(data_level1)
    predictions_l1 = model_level1.predict(data_level1.get_X_test())
    model_level1.print_results(data_level1)
    
    results['Level1'] = {'predictions': predictions_l1, 'true_labels': data_level1.get_type_y_test()}
    
    # Level 2: Type3 prediction for each Type2 class
    print("\nLevel 2 - Type3 for each Type2")
    print("-" * 50)
    
    level2_models = {}
    type2_classes = data_level1.get_classes()
    
    for type2_class in type2_classes:
        # Filter data for this Type2 class
        df_filtered = df[df['y2'] == type2_class].copy()
        
        if len(df_filtered) < 10:  # Skip if too few samples
            continue
        
        # Create index mapping to X
        indices = df_filtered.index.tolist()
        if len(indices) == 0:
            continue
        
        # Get corresponding embeddings
        X_filtered = X[df_filtered.index.tolist()]
        
        df_filtered['y'] = df_filtered['y3']
        data_level2 = Data(X_filtered, df_filtered, label_col='y')
        
        if data_level2.X_train is None:
            continue
        
        print(f"\nType3 model for Type2={type2_class}")
        model_level2 = RandomForest(f"RandomForest_Type3_{type2_class}", data_level2.get_embeddings(), data_level2.get_type())
        model_level2.train(data_level2)
        predictions_l2 = model_level2.predict(data_level2.get_X_test())
        model_level2.print_results(data_level2)
        
        level2_models[type2_class] = {'predictions': predictions_l2, 'true_labels': data_level2.get_type_y_test()}
    
    results['Level2'] = level2_models
    
    # Level 3: Type4 prediction for each Type3 class
    print("\nLevel 3 - Type4 for each Type3")
    print("-" * 50)
    
    level3_models = {}
    
    for type2_class, level2_info in level2_models.items():
        df_l2 = df[df['y2'] == type2_class].copy()
        type3_classes = df_l2['y3'].unique()
        
        for type3_class in type3_classes:
            df_l3 = df_l2[df_l2['y3'] == type3_class].copy()
            
            if len(df_l3) < 5:  # Skip if too few samples
                continue
            
            X_l3 = X[df_l3.index.tolist()]
            df_l3['y'] = df_l3['y4']
            data_level3 = Data(X_l3, df_l3, label_col='y')
            
            if data_level3.X_train is None:
                continue
            
            print(f"\nType4 model for Type2={type2_class}, Type3={type3_class}")
            model_level3 = RandomForest(f"RandomForest_Type4_{type2_class}_{type3_class}", data_level3.get_embeddings(), data_level3.get_type())
            model_level3.train(data_level3)
            predictions_l3 = model_level3.predict(data_level3.get_X_test())
            model_level3.print_results(data_level3)
            
            level3_models[f"{type2_class}_{type3_class}"] = {'predictions': predictions_l3, 'true_labels': data_level3.get_type_y_test()}
    
    results['Level3'] = level3_models
    
    return results


def model_predict(data, df, name):
    """
    Legacy interface for single model prediction.
    Kept for backward compatibility with existing workflows.
    
    Args:
        data: Data object
        df: Dataframe
        name: Group name
    """
    model_predict_single(data)


def model_evaluate(model, data):
    """
    Evaluate model results.
    
    Args:
        model: Model instance
        data: Data object
    """
    model.print_results(data)
