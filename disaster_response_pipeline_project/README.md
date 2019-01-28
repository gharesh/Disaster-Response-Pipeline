# Disaster Response Pipeline Project

## Project Overview
This project holds repository that contains code for application which can be used by employees during a disaster event (e.g. an earthquake or hurricane), to be able to classify the messages into several categories, in order that the message can be directed to the appropriate aid agencies.

The app uses a ML model to categorize any new messages received, and the repository also contains the code used to train the model and to prepare any new datasets for model training purposes.

## File Descriptions
process_data.py: This code takes as its input csv files containing message data and message categories (labels), and creates an SQLite database containing a merged and cleaned version of this data.
train_classifier.py: This code takes the SQLite database produced by process_data.py as an input and uses the data contained within it to train and tune a ML model for categorizing messages. The output is a pickle file containing the fitted model. Test evaluation metrics are also printed as part of the training process.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/model.p`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
