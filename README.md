# Project 2 of Data Scientist course on Udacity: 
# Disaster Response Pipeline
This Pipeline provides a html application (website) where a message related to a crisis can be written. The  with an underlying python code of the application then classifies the message automatically into one or more of the 36 categories. The underlying machine learning algorithm was trained on a collection of categorized disaster messages from [Appen](https://www.appen.com/).

## Installation
The code was written in Python 3.

## Running the application (original instructions from Udacity:)
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Go to http://0.0.0.0:3001/

## Background
The project is the second of four projects in the Data Scientist course of Udacity.

--------------------------------------------------------------

## File folders and content

#### data
Contains the training data from Appen and the code that cleans the data and saves it to an sqlite database. The sqlite database is then ready for the machine learning code in the folder "models".

#### models
The python code in this folder uses a machine learning pipeline with a natural language toolkit to train a model to categorize a disaster message. Various models are tried via GridSearchCV and the best model is saved to a pickle file. the pickle file is not contained in this github repo due to its size of around 100 MB. The pickle file will be saved on your computer if you run the code locally. The pickle file will then be used by the python and html website code in the folder "app".

#### app
The python and html code files load the data and the trained model and display the webpage. In the webpage, an entered message will be classified. the classification among the 36 categories will be schon below with highlighted bars (compare screenshots below).

--------------------------------------------------------------

### Screenshots
![](https://github.com/Ottolio/DisasterResponse/blob/main/pic1.png)
![](https://github.com/Ottolio/DisasterResponse/blob/main/pic2.png)

## Licensing
The code is licensed under MIT and Udacity license. 
The categorized messages data under Appen.
