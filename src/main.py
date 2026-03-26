from preprocess import *
from embeddings import *
from modelling.modelling import chained_multi_output_classification, hierarchical_modelling, model_predict
from modelling.data_model import *
import random

seed = 0
random.seed(seed)
np.random.seed(seed)


def load_data():
    """Load input data from CSV files."""
    df = get_input_data()
    return df


def preprocess_data(df):
    """
    Preprocess data by:
    - De-duplicating content
    - Removing noise
    - Converting to unicode strings
    """
    # De-duplicate input data
    df = de_duplication(df)
    # Remove noise in input data
    df = noise_remover(df)
    # Ensure proper data types
    df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')
    df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
    return df


def get_embeddings(df: pd.DataFrame):
    """
    Convert text to TF-IDF embeddings.
    
    Args:
        df: Dataframe with text columns
        
    Returns:
        Feature embeddings matrix and modified dataframe
    """
    X = get_tfidf_embd(df)  # Get TF-IDF embeddings
    return X, df


def get_data_object(X: np.ndarray, df: pd.DataFrame, label_col: str = 'y'):
    """
    Create Data object that encapsulates model inputs.
    
    Args:
        X: Feature embeddings
        df: Dataframe with labels
        label_col: Column to use as target label
        
    Returns:
        Data object with X_train, X_test, y_train, y_test
    """
    return Data(X, df, label_col=label_col)


def perform_legacy_modelling(data: Data, df: pd.DataFrame, name):
    """
    Perform legacy single-label classification.
    
    Args:
        data: Data object
        df: Dataframe
        name: Group name
    """
    model_predict(data, df, name)


def main():
    """
    Main controller orchestrating the entire workflow.
    Performs both design strategies:
    1. Chained Multi-Output Classification
    2. Hierarchical Modelling
    """
    print("\nModular Multi-Label Email Classification")
    print("=" * 50)
    
    # Step 1: Load data
    print("\nStep 1: Loading data...")
    df = load_data()
    print(f"Records loaded: {df.shape[0]}")
    
    # Step 2: Preprocess data
    print("\nStep 2: Preprocessing data...")
    df = preprocess_data(df)
    print(f"Records after preprocessing: {df.shape[0]}")
    
    # Step 3: Get embeddings
    print("\nStep 3: Generating embeddings...")
    X, df = get_embeddings(df)
    print(f"Embedding shape: {X.shape}")
    
    # Step 4: Run Design Strategy 1 - Chained Multi-Output Classification
    print("\n" + "=" * 50)
    print("Strategy 1: Chained Multi-Output Classification")
    print("=" * 50)
    chained_results = chained_multi_output_classification(df.copy(), X)
    
    # Step 5: Run Design Strategy 2 - Hierarchical Modelling
    print("\n" + "=" * 50)
    print("Strategy 2: Hierarchical Modelling")
    print("=" * 50)
    hierarchical_results = hierarchical_modelling(df.copy(), X)
    
    print("\n" + "=" * 50)
    print("Classification Complete")
    print("=" * 50)


if __name__ == '__main__':
    main()
