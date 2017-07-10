import csv
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import numpy
import random
import pandas
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.utils import shuffle
from sklearn.ensemble import AdaBoostClassifier
import pprint

###################################################

def readData(f_name):
    try:
        f= open(f_name,'r',encoding='ISO-8859-1')
        file_reader=csv.reader(f,delimiter=',')
        tweet_list=[]
        gl_vocab=[]
        gl_vector=[]
        u=0
        print("in loop", f_name)
        for row in file_reader:
            if(row[1]=='0' or row[1]=='1' or row[1]=='-1'):
                tweet_list.append((int(row[1]),row[0]))
                u=u+1
        print("Number of Tweets",u)

        f.close()
        random.shuffle(tweet_list)
        cleaned_tweets= clean(tweet_list)
    except e:
        print(e)
    return cleaned_tweets
    

def main():
    
    file_list = ['Obama.csv', 'Romney.csv','testing-Obama.csv', 'testing-Romney.csv']
    train_Obama=readData(file_list[0])
    test_Obama=readData(file_list[2])
    train_Romney=readData(file_list[1])
    test_Romney=readData(file_list[3])

    ##########################Training ######################################

    obama_result = tweets_classifer(train_Obama, test_Obama)
    print("Obama Tweets Result \n")
    for item in obama_result:
        print(item)
    romney_result = tweets_classifer(train_Romney, test_Romney)
    print("Romney Tweets Result \n")
    for item in romney_result:
        print(item)
    
def tweets_classifer(training_data, testing_data):
    train_class,train_data = zip(*training_data)
    train_class=list(train_class)
    train_data=list(train_data)

    test_class,test_data = zip(*testing_data)
    test_class=list(test_class)
    test_data=list(test_data)

    precision_pos = 0.0
    precision_neg = 0.0
    recall_pos = 0.0
    recall_neg = 0.0
    fscore_pos = 0.0
    fscore_neg = 0.0
    overall_accuracy = 0.0
    
    classifiers = ['LogisticReg', 'LinearSVC','MultiNominalNB', 'SVC', 'AdaBoostClassifier', 'SDGClassifier']
    final_result=[]
    
    for clsfName in classifiers:                   
        regr = make_classifier(clsfName)
        regr.fit(train_data,train_class)
        prediction = regr.predict(test_data);

        precision = precision_score(test_class, prediction, labels =[1, -1, 0], average=None)
        recall = recall_score(test_class, prediction, labels =[1, -1, 0], average=None)
        f1score = f1_score(test_class, prediction, labels =[1, -1, 0], average=None)
        accuracy = accuracy_score(test_class, prediction)
        precision_pos = precision[0]
        precision_neg = precision[1]
        recall_pos = recall[0]
        recall_neg = recall[1]
        fscore_pos = f1score[0]
        fscore_neg = f1score[1]
        overall_accuracy = accuracy
        clsf_result1 = {'Classifier': clsfName, 'Class': 'Positive' ,'precision':(precision_pos), 'recall':(recall_pos), 'f1score':(fscore_pos), 'Overall Accuracy': (overall_accuracy)}
        clsf_result2 = {'Classifier': clsfName, 'Class': 'Negative' ,'precision':(precision_neg), 'recall':(recall_neg), 'f1score':(fscore_neg), 'Overall Accuracy': (overall_accuracy)}
        final_result.append(clsf_result1)
        final_result.append(clsf_result2)
    return final_result


def make_classifier(clsfName):
    txt_clsf_pip = []
    txt_clsf_pip.append(('vect', TfidfVectorizer(analyzer="word", ngram_range=(1,3), stop_words=None, sublinear_tf=True, use_idf = True)))
    txt_clsf_pip.append(('transformer', TfidfTransformer()))
    if clsfName == 'LinearSVC':
        txt_clsf_pip.append(('clsf', LinearSVC()))
    elif clsfName == 'LogisticReg':
        txt_clsf_pip.append(('clsf', LogisticRegression()))
    elif clsfName == 'MultiNominalNB':
        txt_clsf_pip.append(('clsf', MultinomialNB()))
    elif clsfName == 'SVC':
        txt_clsf_pip.append(('clsf', NuSVC(kernel="linear", nu=0.5)))
    elif clsfName == 'AdaBoostClassifier':
        txt_clsf_pip.append(('clsf', AdaBoostClassifier(LogisticRegression(), n_estimators=500, learning_rate=1, algorithm='SAMME', random_state=42)))
    elif clsfName == 'SDGClassifier':
       txt_clsf_pip.append(('clsf', SGDClassifier()))
    return Pipeline(txt_clsf_pip)

def clean(cpy_tweet_list):
    clean_list=[]
    u=0
    for row in cpy_tweet_list:
        u=u+1
        a_tweet=row[1].lower()

        item=re.sub(r"\b(www.|http?)\S+\b","URL", a_tweet) #removing URLs
        
        item=re.sub(r"\S+@\S+","URL", item) #removing emails
        
        item=re.sub(r"(^|\s)[0-9]+[^\s]*","",item) #removing words starting with number
        
        item=re.sub(r"(^|\s)#"," ",item) #removing # from words starting with #
    
        item=re.sub(r"(^|\s)@\S+[^\s]","atusername ",item) #replacing words starting with @ with atusername
        
        #item=re.sub("[!@#$%^&*()[]{};:\,/<>?\|`~-=_+]|\.", " ", item) #removing punctuation and special characters
        item=re.sub("[^a-z0-9\s]", " ", item)
        #print("6------>",item)
    
        cl_item=furtherClean(item) #stripping recurring characters in a word to just two
        
        cl_item=remNLTKStopWords(cl_item) #removing stopwords using NLTK stopwords list
        
        cl_item=porterNLTK(cl_item) #applying stemming
        
        cl_item=re.sub(r'\s+'," ",cl_item)

        cl_item=remNLTKStopWords(cl_item)
        
        clean_list.append((row[0],cl_item.strip()))
    print('cleaned',u)
    return clean_list



def remNLTKStopWords(item):
    sw_list = stopwords.words('english')
    sw_list.append('e')
    for word in sw_list:
        item=re.sub(r"\s\b"+word+r"'"," ",item)
    for word in sw_list:
        item=re.sub(r"\b"+word+r"\b"," ",item)
    return item



def furtherClean(tweet):
    a_tweet=""
    for word in tweet.split():
        li=[]
        uniq=[]
        new_word=""
        for i in range(0,len(word.strip())):
            li.append(word[i])
            uniq.append(word[i])
        for i in range(2,len(li)):
            if(li[i]==li[i-1] and li[i]==li[i-2]):
                uniq.remove(li[i])
        for item in uniq:
            new_word=new_word+item
        a_tweet=a_tweet+new_word+" "   
    return a_tweet.strip()



def porterNLTK(item):
    new_item=""
    p_stemmer=SnowballStemmer('english')
    for word in item.split():
        new_item=new_item+p_stemmer.stem(word)+" "
    return new_item.strip()

if __name__ == '__main__':
    main()
