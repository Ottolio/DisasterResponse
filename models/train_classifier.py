import pandas as pd
import time
import re                  # for removing regular expressions like punctuation
import nltk
nltk.download('punkt')     # tokenization
nltk.download('wordnet')   # download for lemmatization
nltk.download('stopwords') 

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords # from nltk.stem.porter import PorterStemmer ## not used, lemmatization instead
from nltk.stem.wordnet import WordNetLemmatizer
from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer  # for pipeline building
from sklearn.ensemble import RandomForestClassifier                             ## for pipeline building
from sklearn.multioutput import MultiOutputClassifier 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report  # from sklearn.metrics import confusion_matrix   # not used. will use classification report      # will then be plotted to evaluate prediction

import sys             # for filepath
import pickle          # for saving the pickle file

def load_data(database_filepath):
    """     Loads the "Responses_cleaned" table from the database which was generated in "process_data.py" from the disaster_messages.csv and 
            disaster_categories.csv
    Input:  the file_path of the .db file
    Output: X: df with the only the messages
            Y: df with the categories as columns and 0 and 1 values
            Y.keys(): list of category_names 
    """
    engine = create_engine('sqlite:///' + database_filepath)         # for notebook     #start1=time.time()
    df = pd.read_sql_table('Responses_Cleaned', engine)
    X = df['message']  
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)  # axis =1 means dropping columns (axis = 0 --> dropping rows)
    category_names = Y.keys()             # print('Duration: {} seconds'.format(time.time()-start1))  #df.head(1) #X.head(1) #X.describe() #Y.head(1) # X[0] 
                                          # category_names
    return X, Y, category_names
        
def tokenize(text):
    """   to make the natural language text string better machine readable, the follwoing processing steps are done:
        -  removing punctuation
        -  capital letters changed to small, lower case letters
        -  breaking up Sentence into single words
        -  removing words which do not carry much meaning alone (stop words)
        -  changing words to their root form (instead of cutting word words to their stem)
        -  changing words due to their recognized "parts of speech"
    Input: text: one message (not a list) as string
    Output:lemmed: a list of the processed words ready for the machine learning algorithm   
    """
                                                # 'Weather update - a cold front from Cuba that could pass over Haiti'
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
                                                # Remove punctuation characters and  make all lower case
                                                # 'Weather update   a cold front from Cuba that could pass over Haiti'
                                                # 'weather update   a cold front from cuba that could pass over haiti'
    words = word_tokenize(text)                 # break up Sentence in single words, similar effect:    words = text.split()
    words = [w for w in words if w not in stopwords.words("english")]  # Remove stop words -- See: print(stopwords.words("english"))
                                                # ['weather', 'update', 'cold', 'front', 'cuba', 'could', 'pass', 'haiti'])
                                                # stemmed = [PorterStemmer().stem(w) for w in words] # Reduce words to their stems ## not used --> instead lemmatization
                                                # ['weather', 'updat', 'cold', 'front', 'cuba', 'could', 'pass', 'haiti']
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]            # Reduce words to their root form
                                                # ['weather', 'update', 'cold', 'front', 'cuba', 'could', 'pas', 'haiti']
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]      # Lemmatize verbs by specifying part of speech of the word
    return lemmed                               # in notebook test via: tokenize(X[0])


def build_model():
    """ a machine learning pipline with the library sklearn.pipeline.Pipeline
        using two transformers "CountVectorizer" and "TfidfTransformer" and "MultiOutputClassifier".
        Optional parameters outcommented to set Gridsearch so that pipeline does not crash on laptop and in jupyter notebook and to keep computation time shorter
        
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',  MultiOutputClassifier(RandomForestClassifier()))  #if n_estimators is set as high number then computation takes too long, therefore set below    
    ])                                            # https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html
    # parameters = { 'clf__estimator__n_estimators': [5,10] }    # Values tried due to Udacity Q&A from Ankit / Rajat S. --> use if it takes too long 
    #  pipeline = GridSearchCV(pipeline, param_grid=parameters)
    
    return pipeline
    
def evaluate_model(model, X_test, Y_test, category_names):
    """ predicting the categories of the messages and calculating the "precision", "recall" and "f1 score" 
    to evaluate how good the prediction is in comparision to the testing data.
    Input: model: the build model from "model_build"
        X_test: messages from the testing dataset (that was split from the original messages data)
        Y_test: Categories of the test messages as refernce to compare the predictions to.
        category_names: the different categories (column names of the y-data)
    Output: no output due to the "pass" command as output is very long.
    """
                                                # start3=time.time()
    y_pred = model.predict(X_test)           # print('Duration: {} seconds'.format(time.time()-start3))
                                                # start4=time.time()
           #print(classification_report()) # display_results(y_test, y_pred) does not work in notebook, needs to be done column by column
           # Udacity Q&A from Mike/Rajat and git hub https://github.com/MikeDurrant/DisasterResponse/blob/master/web_app/models/train_classifier.py
    n=0
    for col in Y_test.columns:
            print("Column_tested:{}".format(Y_test.columns[n]))
            print("Result:")
            print(classification_report(Y_test.iloc[:,n], y_pred[:,n]))
            n+=1
                                                   # print('Duration: {} seconds'.format(time.time()-start4))
    pass

def save_model(model, model_filepath):
    """    saving model to pickle file with "Pickle"
    Input: the model and its file location
    """
    temp = open(model_filepath,'wb')
    pickle.dump(model, temp)
    temp.close()
    pass

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