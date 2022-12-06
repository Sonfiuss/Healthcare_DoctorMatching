import pandas as pd
import numpy as np
import warnings
import nltk
import string
import re
import gensim
import pickle

from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn 
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from gensim.models import Word2Vec


# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

warnings.filterwarnings('ignore')
# Load raw data
df = pd.read_csv('data_csv/overview-of-recordings.csv')

duplicate=df.duplicated().sum()
# collection the texts needed
Text = df[['phrase', 'prompt']]
# save English stopwords
stopwords_list = set(stopwords.words("english"))
def phrase_cleanse(phrase):
    #Tokenize and divide phrase into separate words
    token_words = word_tokenize(phrase)
    
    # Convert all texts to lower cases
    words_step1 = []
    for word_1 in token_words:
        words_step1.append(word_1.lower())
    
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
text = np.array(Text.loc[:,'phrase'])
new_text = []
for i in text:
    new_text.append(phrase_cleanse(i))
Text.insert(2,'new_text',new_text)

text_vectorize = TfidfVectorizer()
X_tf_idf = text_vectorize.fit_transform(Text["new_text"])

dense_list = X_tf_idf.todense().tolist()
feature_names = text_vectorize.get_feature_names()
df_tf_idf = pd.DataFrame(dense_list, columns = feature_names)

# concatenate prompt column with tf_idf matrix
text_tf_idf = pd.concat([Text["prompt"], df_tf_idf], axis = 1)

n = Text['prompt'].nunique()
text_hashvectorize = HashingVectorizer(n_features = n*3)
X_hash = text_hashvectorize.fit_transform(Text["new_text"])

df_hash_vectorize = pd.DataFrame(X_hash.toarray())

# concatenate prompt column with hash vectorized matrix
text_hash_vectorize = pd.concat([Text["prompt"], df_hash_vectorize], axis = 1)
bag_word = CountVectorizer()
feature_bow = bag_word.fit_transform(Text["new_text"].values)

# maping feature 
df_bow = pd.DataFrame(feature_bow.todense().tolist(), columns = bag_word.get_feature_names())

# concatenate prompt column with bow matrix
bag_word_df = pd.concat([Text['prompt'], df_bow], axis = 1)
bag_word_df.to_csv('data_csv/bag_word_df.csv',index=False)
Text['new_text_clean'] = Text['new_text'].apply(lambda x: x.split(" "))

# Train the word2vec model
w2v_model = Word2Vec(Text['new_text_clean'], vector_size=100, window=5, min_count=1, workers=4)

# Take the average of the word vectors for the words contained in each sentence
def word_avg_vect(data, model, num_features):
    words = set(model.wv.index_to_key)
    X_vect = np.array([np.array([model.wv[i] for i in s if i in words]) for s in data])
    X_vect_avg = []
    for v in X_vect:
        if v.size:
            X_vect_avg.append(v.mean(axis = 0))
        else:
            X_vect_avg.append(np.zeros(num_features, dtype = float))

    df_vect_avg = pd.DataFrame(X_vect_avg)
    return df_vect_avg

X_w2v = word_avg_vect(Text['new_text_clean'], w2v_model, 100)
# concatenate prompt column with averaged w2v matrix
df_w2v = pd.concat([Text["prompt"], X_w2v], axis = 1)
df_w2v.to_csv(f"data_csv/df_w2v.csv", index=False)

def PCA_project(data, data_name="", threshold = 99):
    max_component = data.shape[1]
    cutoff = threshold
    covar_matrix = PCA(n_components = max_component)
    covar_matrix.fit(data)
    variance = covar_matrix.explained_variance_ratio_
    var = np.cumsum(np.round(variance, decimals = 4)*100)
    index = 0
    for i in range(len(var)):
        
        if np.round(var[i]) < cutoff:
            index += 1
        else:
            break
    principal=PCA(n_components=index)
    principal.fit(data)
    pickle.dump(principal, open(data_name+'.pkl','wb'))
    print('%s reduce features from %d to %d'% (data_name,max_component, index))
    return principal.transform(data)

bow_P= PCA_project(df_bow, 'bag of words')
tf_idf_P= PCA_project(df_tf_idf, 'tf_idf')
hash_P= PCA_project(df_hash_vectorize, 'hash_vectorize')
w2v_P= PCA_project(np.array(X_w2v), 'word2vec')

y = Text["prompt"]
X_bow_train, X_bow_test, y_bow_train, y_bow_test = train_test_split(bow_P,y,test_size = 0.2, random_state =3, stratify = y)
X_tf_train, X_tf_test, y_tf_train, y_tf_test = train_test_split(tf_idf_P,y,test_size = 0.2, random_state =3, stratify = y)
X_hash_train, X_hash_test, y_hash_train, y_hash_test = train_test_split(hash_P,y,test_size = 0.2, random_state =3, stratify = y)
X_w2v_train, X_w2v_test, y_w2v_train, y_w2v_test = train_test_split(w2v_P,y,test_size = 0.2, random_state =3, stratify = y)

def matric_table(model_list, name_list,y_data, X_data):
    result = []
    for m,n,a,b in zip(model_list, name_list, y_data, X_data):
        report = []
        report.append(n)
        report.append(accuracy_score(a[0], m.predict(b[0])) * 100)
        report.append(accuracy_score(a[1], m.predict(b[1])) * 100)
        report.append(recall_score(a[1], m.predict(b[1]),average = 'weighted') * 100)
        report.append(precision_score(a[1], m.predict(b[1]),average = 'weighted') * 100)
        report.append(f1_score(a[1], m.predict(b[1]),average = 'weighted') * 100)
        result.append(report)
    df = pd.DataFrame(data = result, columns=['Model', 'Training Accuracy %', 'Testing Accuracy %','Testing precision %', 'Testing recall %', 'Testing f1_score %'])
    df = df.set_index('Model')
    return df.style.highlight_max(color = 'lightgreen', axis = 0)

rf_bow = RandomForestClassifier(n_estimators=100,
                                max_features=None,
                                oob_score=True,
                                n_jobs=-1,
                                random_state=0)
rf_tf = RandomForestClassifier(n_estimators=100,
                                max_features=None,
                                oob_score=True,
                                n_jobs=-1,
                                random_state=0)
rf_hash = RandomForestClassifier(n_estimators=100,
                                max_features=None,
                                oob_score=True,
                                n_jobs=-1,
                                random_state=0)
rf_w2v = RandomForestClassifier(n_estimators=100,
                                max_features=None,
                                oob_score=True,
                                n_jobs=-1,
                                random_state=0)

rf_bow.fit(X_bow_train, y_bow_train)
rf_tf.fit(X_tf_train, y_tf_train)
rf_hash.fit(X_hash_train, y_hash_train)
rf_w2v.fit(X_w2v_train, y_w2v_train)

model_list = [rf_bow,rf_tf,rf_hash,rf_w2v]
name_list = ["Random Forest with bow","Random Forest with tf_idf", "Random Forest with hash","Random Forest with word2vec"]
y_data = [[y_bow_train,y_bow_test], [y_tf_train,y_tf_test], [y_hash_train,y_hash_test],[y_w2v_train,y_w2v_test]]
X_data = [[X_bow_train,X_bow_test], [X_tf_train,X_tf_test], [X_hash_train,X_hash_test],[X_w2v_train,X_w2v_test]]
matric_table(model_list, name_list, y_data, X_data)

lr_bow = LogisticRegression()
lr_tf = LogisticRegression()
lr_hash = LogisticRegression()
lr_w2v = LogisticRegression()

lr_bow.fit(X_bow_train, y_bow_train)
lr_tf.fit(X_tf_train, y_tf_train)
lr_hash.fit(X_hash_train, y_hash_train)
lr_w2v.fit(X_w2v_train, y_w2v_train)

model_list = [lr_bow,lr_tf,lr_hash,lr_w2v]
name_list = ["Logistic Regression with bow","Logistic Regression with tf_idf", "Logistic Regression with hash","Logistic Regressiont with word2vec"]
y_data = [[y_bow_train,y_bow_test], [y_tf_train,y_tf_test], [y_hash_train,y_hash_test],[y_w2v_train,y_w2v_test]]
X_data = [[X_bow_train,X_bow_test], [X_tf_train,X_tf_test], [X_hash_train,X_hash_test],[X_w2v_train,X_w2v_test]]
matric_table(model_list, name_list, y_data, X_data)



svc_bow = SVC(decision_function_shape='ovr')
svc_tf = SVC(decision_function_shape='ovr')
svc_hash = SVC(decision_function_shape='ovr')
svc_w2v = SVC(decision_function_shape='ovr')

svc_bow.fit(X_bow_train, y_bow_train)
svc_tf.fit(X_tf_train, y_tf_train)
svc_hash.fit(X_hash_train, y_hash_train)
svc_w2v.fit(X_w2v_train, y_w2v_train)

# model_list = [svc_bow,svc_tf,svc_hash,svc_w2v]
# name_list = ["SVC with bow","SVC with tf_idf", "SVC with hash","SVC with word2vec"]
# y_data = [[y_bow_train,y_bow_test], [y_tf_train,y_tf_test], [y_hash_train,y_hash_test],[y_w2v_train,y_w2v_test]]
# X_data = [[X_bow_train,X_bow_test], [X_tf_train,X_tf_test], [X_hash_train,X_hash_test],[X_w2v_train,X_w2v_test]]
# matric_table(model_list, name_list, y_data, X_data)
knn_bow = KNeighborsClassifier(n_neighbors=3)
knn_tf = KNeighborsClassifier(n_neighbors=3)
knn_hash = KNeighborsClassifier(n_neighbors=3)
knn_w2v = KNeighborsClassifier(n_neighbors=3)

knn_bow.fit(X_bow_train, y_bow_train)
knn_tf.fit(X_tf_train, y_tf_train)
knn_hash.fit(X_hash_train, y_hash_train)
knn_w2v.fit(X_w2v_train, y_w2v_train)

# model_list = [knn_bow,knn_tf,knn_hash,knn_w2v]
# name_list = ["KNN with bow","KNN with tf_idf", "KNN with hash","KNN with word2vec"]
# y_data = [[y_bow_train,y_bow_test], [y_tf_train,y_tf_test], [y_hash_train,y_hash_test],[y_w2v_train,y_w2v_test]]
# X_data = [[X_bow_train,X_bow_test], [X_tf_train,X_tf_test], [X_hash_train,X_hash_test],[X_w2v_train,X_w2v_test]]
# matric_table(model_list, name_list, y_data, X_data)

gnb_bow = GaussianNB()
gnb_tf = GaussianNB()
gnb_hash = GaussianNB()
gnb_w2v = GaussianNB()

gnb_bow.fit(X_bow_train, y_bow_train)
gnb_tf.fit(X_tf_train, y_tf_train)
gnb_hash.fit(X_hash_train, y_hash_train)
gnb_w2v.fit(X_w2v_train, y_w2v_train)

# model_list = [gnb_bow,gnb_tf,gnb_hash,gnb_w2v]
# name_list = ["Gaussian Naive Bayes with bow","Gaussian Naive Bayes with tf_idf", "Gaussian Naive Bayes with hash","Gaussian Naive Bayes with word2vec"]
# y_data = [[y_bow_train,y_bow_test], [y_tf_train,y_tf_test], [y_hash_train,y_hash_test],[y_w2v_train,y_w2v_test]]
# X_data = [[X_bow_train,X_bow_test], [X_tf_train,X_tf_test], [X_hash_train,X_hash_test],[X_w2v_train,X_w2v_test]]
# matric_table(model_list, name_list, y_data, X_data)

# model_list = [rf_w2v,lr_bow,svc_tf,knn_bow,gnb_tf]
# name_list = ["Random Forest with w2v","Logistic Regression with bow", "SVC with tf","KNN withbow","gaussian Naive Bayes with tf"]
# y_data = [[y_w2v_train,y_w2v_test], [y_bow_train,y_bow_test], [y_tf_train,y_tf_test],[y_bow_train,y_bow_test],[y_tf_train,y_tf_test]]
# X_data = [[X_w2v_train,X_w2v_test], [X_bow_train,X_bow_test], [X_tf_train,X_tf_test],[X_bow_train,X_bow_test],[X_tf_train,X_tf_test]]
# matric_table(model_list, name_list, y_data, X_data)

best_models = []
n_neighbors = [3,5,7,9]
weights = ['uniform','distance']
ps = [1,2]

def KNN_clf(n_neighbors, weight, p):
    knn = KNeighborsClassifier(n_neighbors = n_neighbors, weights = weight,p = p)
    knn.fit(X_bow_train, y_bow_train)
    y_pred = knn.predict(X_bow_test)
    n = accuracy_score(y_bow_test,y_pred)
    best_models.append((n_neighbors, weight, p ,n))

for c in n_neighbors:
    for w in weights:
        for p in ps:
            KNN_clf(c, w, p)

print(max(best_models,key=lambda item:item[3]))


best_models = []
crit = ['gini', 'entropy']
max_d = range(1,20,4)
min_s_leaf = range(1,20,4)
n_est = [50, 100, 200]

def RF_clf(crit, max_d, min_s_leaf, n_est):
    forest = RandomForestClassifier(criterion=crit, max_depth=max_d, min_samples_leaf=min_s_leaf, n_estimators=n_est, random_state=1)
    forest.fit(X_w2v_train, y_w2v_train)
    y_pred = forest.predict(X_w2v_test)
    n = accuracy_score(y_w2v_test,y_pred)
    best_models.append((crit,max_d,min_s_leaf,n_est,n))


for c in crit:
    for md in max_d:
        for msl in min_s_leaf:
            for n_e in n_est:
                RF_clf(c, md, msl, n_e)

print(max(best_models,key=lambda item:item[4]))

Knn_best = KNeighborsClassifier(n_neighbors = 3, weights = 'uniform' ,p = 1)
Rf_best = RandomForestClassifier(criterion='gini', max_depth=13, min_samples_leaf=1, n_estimators=50, random_state=1)
Knn_best.fit(X_bow_train, y_bow_train)
Rf_best.fit(X_w2v_train, y_w2v_train)

# model_list = [Knn_best, Rf_best]
# name_list = ["Tuned KNN", 'Tuned Randome Forest']
# y_data = [[y_bow_train,y_bow_test], [y_w2v_train,y_w2v_test] ]
# X_data = [[X_bow_train,X_bow_test], [X_w2v_train,X_w2v_test]]
# matric_table(model_list, name_list, y_data, X_data)

pickle.dump(Rf_best, open('bestmodel.pkl','wb'))
pickle.dump(w2v_model, open('w2v_model.pkl','wb'))

raw_input = input()

def input_process(data):
    input_clean = phrase_cleanse(data)
    w2v_model = pickle.load(open('w2v_model.pkl', 'rb'))
    input_clean = [input_clean.split(" ")]
    processed_input = word_avg_vect(input_clean, w2v_model, 100)
    pca_model = pickle.load(open('word2vec.pkl', 'rb')) 
    test = pca_model.transform(processed_input)
    return test

def pred(data):
    test = input_process(data)
    model = pickle.load(open('bestmodel.pkl', 'rb'))
    prediction = model.predict(test)
    return prediction

pred(raw_input)
