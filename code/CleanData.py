import pandas as pd
import warnings
import numpy as np
import string
import re

import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn 
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')


class CleanData:

    def __init__(self, df, Text, duplicate):
        self.df = df
        self.Text = Text
        self.duplicate = duplicate
    
    def read_csv(self):
        print("=================")
        warnings.filterwarnings('ignore')
        # Load raw data
        # df = pd.read_csv('data_csv\overview-of-recordings.csv')
        # df.info()
        df = pd.read_csv('data_csv\overview-of-recordings.csv')
        #start cleansing
        #count duplicate
        self.duplicate= self.df.duplicated().sum()
        # collection the texts needed
        self.Text = df[['phrase', 'prompt']]
        stopwords_list = set(stopwords.words("english"))
        return stopwords_list

    def phrase_cleanse(phrase):
        print("+++++++++++++++++++++")
        #Tokenize and divide phrase into separate words
        token_words = word_tokenize(phrase)
        
        # Convert all texts to lower cases
        words_step1 = []
        for word_1 in token_words:
            words_step1.append(word_1.lower())
        #stopwords_list
        stopwords_list = CleanData.read_csv()
        
        #Clear all punctuation
        words_step2 = [] 
        for word_2 in words_step1:
            word_cleaned = re.sub(r'[^\w\s]','',word_2)
            words_step2.append(word_cleaned)
        
        #Clean the text list
        words_step3 = []
        for word_3 in words_step2:
            # check if every characters are alphbets
            if word_3.isalpha():
                # get rid of stop words
                if word_3 not in list(stopwords_list):
                    words_step3.append(word_3)
                else:
                    continue
        
        #Lemmatization - group different forms of same word which has more than 2 characters into one word
        lem = nltk.stem.WordNetLemmatizer()
        lem_list = []
        for word_4 in words_step3:
            if(len(word_4) > 2):
                lem_list.append(lem.lemmatize(word_4))
        
        join_text = " ".join(lem_list)
        
        return join_text
    def new_text(self):
        print("-----------------")
        text = np.array(self.Text.loc[:,'phrase'])
        new_text = []
        for i in text:
            new_text.append(self.phrase_cleanse(i))
        self.Text.insert(2,'new_text',new_text)
        self.Text.to_csv(f"datacsv/cleaned_data.csv", index=False)
        return self.Text

test = CleanData()
print(test.new_text())