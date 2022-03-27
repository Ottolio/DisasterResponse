# Project 2 of Data Scientist course on Udacity: 
# Disaster Response Pipeline
This Pipeline provides a thml application (website) where a message related to a crisis can be written. The application then classifies the message automatically into one or more of the 36 categories. The underlying machine learning algorithm was trained on a collection of categorized disaster messages from [Appen](https://www.appen.com/).

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


#### Screenshots
![](https://github.com/Ottolio/DisasterResponse/edit/main/pic1.png)
![](https://github.com/Ottolio/DisasterResponse/edit/main/pic2.png)

## Licensing
The code is licensed under MIT and Udacity license. 
The categorized messages data under Appen.

