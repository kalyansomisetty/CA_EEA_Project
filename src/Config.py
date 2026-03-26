class Config:
    # Input Columns
    TICKET_SUMMARY = 'Ticket Summary'
    INTERACTION_CONTENT = 'Interaction content'

    # Type Columns to test
    TYPE_COLS = ['y2', 'y3', 'y4']
    CLASS_COL = 'y2'
    GROUPED = 'y1'
    
    # Multi-label Configuration
    MIN_CLASS_COUNT = 3
    TEST_SIZE = 0.2
    RANDOM_STATE = 0
    
    # TF-IDF Parameters
    VECTORIZER_MAX_FEATURES = 2000
    VECTORIZER_MIN_DF = 4
    VECTORIZER_MAX_DF = 0.90
    
    # Model Parameters
    RANDOM_FOREST_N_ESTIMATORS = 1000
    RANDOM_FOREST_CLASS_WEIGHT = 'balanced_subsample'
