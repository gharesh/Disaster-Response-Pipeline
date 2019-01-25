import sys
import nltk
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
import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer


def load_data(database_filepath):
    """
    Load data from SQLite database and split into features and target
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('Message', engine)
    # drop columns with null
    df = df[~(df.isnull().any(axis=1))|((df.original.isnull())&~(df.offer.isnull()))]
        
    X = df['message']
    y = df.iloc[:,4:]
    categories = y.columns

    return X,y,categories

def tokenize(text):
    """
    Remove capitalization and special characters and lemmatize texts
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
    """

    # create pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([('text', Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                                                     ('tfidf', TfidfTransformer()),
                                                     ])),
                                  ('length', Pipeline([('count', FunctionTransformer(length_of_messages, validate=False))]))]
                                 )),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    # use GridSearch to tune model with optimal parameters
    parameters = {'features__text__vect__ngram_range':[(1,2),(2,2)],
            'clf__estimator__n_estimators':[50, 100]
             }
    model = GridSearchCV(pipeline, parameters)

    return model

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Show precision, recall, f1-score of model scored on testing set
    """    
    
    # make predictions with model
    Y_pred = model.predict(X_test)

    # print scores
    print(classification_report(Y_test.iloc[:,1:].values, np.array([x[1:] for x in Y_pred]), 
        target_names=category_names))


def save_model(model, model_filepath):
    """
    Pickle model to designated file
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