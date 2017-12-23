
from __future__ import  division
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
 
 
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
import string
#import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import ensemble, metrics, model_selection, naive_bayes
 

eng_stopwords = set(stopwords.words("english"))
pd.options.mode.chained_assignment = None
## Read the train and test dataset and check the top few lines ##
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
print("Number of rows in train dataset : ",train_df.shape[0])
print("Number of rows in test dataset : ",test_df.shape[0])


def  remove(txt):
    result = ''.join([i for i in txt if not i.isdigit()])
    return result
train_df=train_df.fillna('kinetic')
test_df=test_df.fillna('kinetic')
def  kinetic(row):
    probs=np.unique(row,return_counts=True)[1]/len(row)
    kinetic=np.sum(probs**2)
    return kinetic

# def kinetic_letters(text):
#     text = text.lower()
#     letterRepartition = np.zeros(26)
#     i = 0
#     for letter in text:
#         if ord(letter) in range(97, 123) :
#             letterRepartition[ord(letter)-97] +=1 
#     probs = letterRepartition/len(text)
#     kinetic = np.sum(probs**2)
#     return kinetic
    
def kinetic_letters(text):
 
    letterRepartition = np.zeros(26)
    for letter in text:
        if ord(letter) in range(97, 123) :
            letterRepartition[ord(letter)-97] +=1
    letterRepartition = letterRepartition / len(text)
    return kinetic(letterRepartition)

def kinetic_voals(text):
  
    letterRepartition = np.zeros(26)
    for letter in text:
        if ord(letter) in range(97, 123) :
            letterRepartition[ord(letter)-97] +=1 
            
    letterRepartition = letterRepartition / len(text)       
    return kinetic(letterRepartition[[0, 4, 8, 14, 20, 24]])

def kinetic_cons(text):
   
    letterRepartition = np.zeros(26)
    for letter in text:
        if ord(letter) in range(97, 123) :
            letterRepartition[ord(letter)-97] +=1 
    letterRepartition = letterRepartition / len(text)
    return kinetic(letterRepartition[[1, 2, 3 , 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18 ,19 , 21, 22, 
                                     23, 25]])

def kinetic_ponct(text):
 
    ponct_list = list(['.', ',', ';', '?', '!'])
    ponct_repart = np.zeros(5)
    for letter in text:
        if letter in ponct_list:
            ponct_repart[ponct_list.index(letter)] += 1
    ponct_repart = ponct_repart / len(text)
    return kinetic(ponct_repart)

def kinetic_average_words(text):
   
    ponct_list = list(['.', ',', ';', '?', '!'])
    for ponct in ponct_list:
        text = text.replace(ponct, '')
    text = text.split(' ')
    avg_kin = 0
    for word in text:
        avg_kin += kinetic_letters(word)
    return avg_kin/len(text)
        

print(train_df["comment_text"].apply(kinetic_average_words))


## kinetic in letters
train_df["kinetic_letters"] = train_df["comment_text"].apply(kinetic_letters)
test_df["kinetic_letters"] = test_df["comment_text"].apply(kinetic_letters)

## kinetic in voals
train_df["kinetic_voals"] = train_df["comment_text"].apply(kinetic_voals)
test_df["kinetic_voals"] = test_df["comment_text"].apply(kinetic_voals)

## kinetic in cons
train_df["kinetic_cons"] = train_df["comment_text"].apply(kinetic_cons)
test_df["kinetic_cons"] = test_df["comment_text"].apply(kinetic_cons)

## kinetic in ponct
train_df["kinetic_ponct"] = train_df["comment_text"].apply(kinetic_ponct)
test_df["kinetic_ponct"] = test_df["comment_text"].apply(kinetic_ponct)

## kinetic in ponct
train_df["kinetic_avg_words"] = train_df["comment_text"].apply(kinetic_average_words)
test_df["kinetic_avg_words"] = test_df["comment_text"].apply(kinetic_average_words)

## Number of words in the text ##
train_df["num_words"] = train_df["comment_text"].apply(lambda x: len(str(x).split()))
test_df["num_words"] = test_df["comment_text"].apply(lambda x: len(str(x).split()))

## Number of unique words in the text ##
train_df["num_unique_words"] = train_df["comment_text"].apply(lambda x: len(set(str(x).split())))
test_df["num_unique_words"] = test_df["comment_text"].apply(lambda x: len(set(str(x).split())))


## Number of characters in the text ##
train_df["num_chars"] = train_df["comment_text"].apply(lambda x: len(str(x)))
test_df["num_chars"] = test_df["comment_text"].apply(lambda x: len(str(x)))

## Number of stopwords in the text ##
train_df["num_stopwords"] = train_df["comment_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))
test_df["num_stopwords"] = test_df["comment_text"].apply(lambda x: len([w for w in str(x).lower().split() if w in eng_stopwords]))

## Number of punctuations in the text ##
train_df["num_punctuations"] =train_df['comment_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )
test_df["num_punctuations"] =test_df['comment_text'].apply(lambda x: len([c for c in str(x) if c in string.punctuation]) )

# Number of conconnes in the text ##


## Number of title case words in the text ##
train_df["num_words_upper"] = train_df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))
test_df["num_words_upper"] = test_df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.isupper()]))

## Number of title case words in the text ##
train_df["num_words_title"] = train_df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))
test_df["num_words_title"] = test_df["comment_text"].apply(lambda x: len([w for w in str(x).split() if w.istitle()]))

## Average length of the words in the text ##
train_df["mean_word_len"] = train_df["comment_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))
test_df["mean_word_len"] = test_df["comment_text"].apply(lambda x: np.mean([len(w) for w in str(x).split()]))

features=test_df.columns[1:]
features
from sklearn.model_selection import train_test_split
train_mes, valid_mes, train_l,valid_l = train_test_split(train_df[features],train_df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']], test_size=0.1, random_state=2)
def text_process(comment):
    nopunc = [char for char in comment if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

transform_com = TfidfVectorizer().fit(pd.concat
                                      ([train_df['comment_text'],test_df['comment_text']],axis=0))


comments_train = transform_com.transform(train_mes['comment_text'])
comments_valid = transform_com.transform(valid_mes['comment_text'])
comments_test = transform_com.transform(test_df['comment_text'])


import scipy
comments_train=scipy.sparse.hstack([comments_train,train_mes[features[1:]]])


comments_valid=scipy.sparse.hstack([comments_valid,valid_mes[features[1:]]])




comments_test = scipy.sparse.hstack([comments_test,test_df[features[1:]]])


 

col = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
preds = np.zeros((test_df.shape[0], len(col)))

import  gc
#for i, j in enumerate(col):
#        
#    print('fit '+j)
#    model = runXGB(comments_train, train_l[j], comments_valid,valid_l[j])
#    preds[:,i] = model.predict(xgb.DMatrix(comments_test))
#    gc.collect()
nrow=comments_train.shape[0]

coly = [c for c in train.columns if c not in ['id','comment_text']]
y = train[coly]
test_id = test['id'].values
from sklearn.ensemble  import ExtraTreesClassifier
model = ensemble.ExtraTreesClassifier(n_jobs=-1, random_state=3)



model = ensemble.ExtraTreesClassifier(n_jobs=-1, random_state=3)
model.fit(comments_train,train_l)
preds=model.predict(comments_test)




subm = pd.read_csv('sample_submission.csv')    
submid = pd.DataFrame({'id': subm["id"]})
submission = pd.concat([submid, pd.DataFrame(preds, columns = col)], axis=1)
submission.to_csv('sub_kinetic_forest.csv', index=False)







