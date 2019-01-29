import sys
import nltk
import pickle
nltk.download(['punkt', 'wordnet'])
import warnings
warnings.filterwarnings("ignore")
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer


def load_data(database_filepath):
    """
    Load data from SQLite database and split into features and target
    Args:
    database_filepath: string. Filename for SQLite database containing cleaned message data.
       
    Returns:
    X: dataframe. Dataframe containing features dataset.
    y: dataframe. Dataframe containing labels dataset.
    categories: list of strings. List containing category names.
    """
    # Load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('Message', engine)
    # drop columns with null
    df = df[~(df.isnull().any(axis=1))|((df.original.isnull())&~(df.offer.isnull()))]
    # Create X and y datasets    
    X = df['message']
    y = df.iloc[:,4:]
    categories = y.columns

    return X,y,categories

def tokenize(text):
    """
    Remove capitalization and special characters and lemmatize texts
    
    Args:
    text: string. String containing message for processing
       
    Returns:
    clean_tokens: list of strings. List containing normalized and stemmed word tokens
    """  
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

def length_of_messages(data):
    """
    Compute the character length of texts
    """
    return np.array([len(text) for text in data]).reshape(-1, 1)

def build_model():
    """
    Build model with a pipeline
    Args:
    None
       
    Returns:
    cv: gridsearchcv object. Gridsearchcv object that transforms the data, creates the 
    model object and finds the optimal model parameters.
    """

    # Create pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize, min_df = 5)),
        ('tfidf', TfidfTransformer(use_idf = True)),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators = 10,
                                                             min_samples_split = 10)))
    ])
    
    # Create parameters dictionary
    parameters = {'vect__min_df': [1, 5],
                  'tfidf__use_idf':[True, False],
                  'clf__estimator__n_estimators':[10, 25], 
                  'clf__estimator__min_samples_split':[2, 5, 10]}
    
    
    # Create grid search object
    cv = GridSearchCV(pipeline, param_grid = parameters, verbose = 10)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """Returns classification report for the model
    
    Args:
    model: model object. Fitted model object.
    X_test: dataframe. Dataframe containing test features dataset.
    Y_test: dataframe. Dataframe containing test labels dataset.
    category_names: list of strings. List containing category names.
    
    Returns:
    None
    """ 
    
    # make predictions with model
    Y_pred = model.predict(X_test)

    # print scores
    print(classification_report(Y_test.iloc[:,1:].values, np.array([x[1:] for x in Y_pred]), 
        target_names=category_names))


def save_model(model, model_filepath):
    """
    Pickle model to designated file
    model: model object. Fitted model object.
    model_filepath: string. Filepath for where fitted model should be saved
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
