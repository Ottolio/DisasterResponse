import sys
# import libraries
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# %matplotlib inline 
from sqlalchemy import create_engine   # as in instructions instead of sqlite3 from previous exercises

def load_data(messages_filepath, categories_filepath):
    """      get the data from disaster_categories and disaster_messages COMMA SEPARATED VALUES files
             combine the two datasets with merge with repsct to column "ID"
    Input:    categories_filepath: location of categories, .csv
              messages_filepath: location of messages, .csv
    Output:   combined dataset df
    """
 
    messages = pd.read_csv(messages_filepath)         # load messages dataset    # messages.head(2)
    categories = pd.read_csv(categories_filepath)     # load categories dataset
    df = messages.merge(categories, on='id')         # merge datasets # df.head(2)  #showing only first two rows to save space and make the notebook shorter
    return df
    
def clean_data(df):
    """     Cleaning the merged dataframe from previous loading function
        -   1 Splitting up categories into 36 different columns 
        -   rename the columns of `categories'.
        -   Convert category values to just numbers 0 or 1.
        -   replace old category column in df with new 36 category columns
        -   removing rows that are duplicates
        -   from notebook: analysing result and then
        -   removing values "2" from column related.
    
    Input:  df: Dataframe. Merged Dataframe with messages and categories
    Output: df: Dataframe. Cleaned Datframe with added columns for each categorie
    """  
    categories = df['categories'].str.split(';', expand=True)   # create a dataframe of the 36 individual category columns ##  categories.head(2)
    row = categories.iloc[0]  # select the first row of the categories dataframe     #[0] means locate first row, [1] would be second row

                                                        # use this row to extract a list of new column names for categories.
                                                        # one way is to apply a lambda function that takes everything 
                                                        # up to the second to last character of each string with slicing
    category_colnames =  row.str.split('-').str.get(0)  # str.get(0) means getting first of the two splitted parts ## print(category_colnames)
    categories.columns = category_colnames  # rename the columns of `categories` with the first part that was split    ##    categories.head(2)
    
    for column in categories:   # Convert category values to just numbers 0 or 1.    # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-').str.get(1)  #get(1) means now get second part of split, i.e. the number
        categories[column] = pd.to_numeric(categories[column])          # the number was still a string, now turn into a number  ##  categories.head(2)
    
    df.drop(['categories'], axis=1, inplace=True)  # drop the original categories column from `df`  ## in notebook: if error occurs then click: "Kernel, clear output" and restart & run all in jupyter notebook  ### df.head(1)
    
    df = pd.concat([df, categories], axis=1)  # concatenate the original dataframe with the new `categories` dataframe   ## df.head(1)
    
    #df.drop_duplicates().head(10) #df without duplicates, showing some rows
    # check number of duplicates in notebook: len(df)-len(df.drop_duplicates())   # from https://stackoverflow.com/questions/35584085/how-to-count-duplicate-rows-in-pandas-dataframe  result is 170  # now (really) drop duplicates
    
    df.drop_duplicates(inplace=True) # True means removing rows with duplicates        # df.head(1)
    
    # in notebook: # check number of duplicates again: len(df)-len(df.drop_duplicates())  result is "0", so it works
    # in notebook: analyze the result: print('Shape:', df.shape) and df.describe()    
    # get overview of data by plotting mean category values: df_categories = df.drop('id', 1) ## df_categories.mean().plot() showing high values in column related. make histogram ## df['related'].hist()  # (needs importing matplotlib! at the very top)
    # the plots show a number "2" occuring in column "related", changing it to "1" to make sense
    df.related.replace(2,1,inplace=True)
    
    return df
    
    
def save_data(df, database_filename):
    """    Saving the cleand dataframe in an sql file. with sqlalchemy imported above, not with sqlite as in previous excercises
    Input: df: cleaned responses dataframe
           database_filename: location of data 
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Responses_Cleaned', engine, if_exists='replace', index=False) # added: if_exists='replace',
    # if "index = True" would be stated then: Write DataFrame index as a column. Uses index_label as the column name in the table. 

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()