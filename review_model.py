
# importing the required libraries
import pandas as pd
import numpy as np
import warnings as w
import pickle as p
from glob import glob
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression



w.filterwarnings('ignore')

# dividing the huge dataset into small chunks
df_reader = pd.read_json('Clothing_Shoes_and_Jewelry.json', lines = True, chunksize = 1000000 )

counter = 1
for chunk in df_reader:
    new_df = pd.DataFrame(chunk[['overall','reviewText','summary']])
    new_df1 = new_df[new_df['overall'] == 5].sample(4000)
    new_df2 = new_df[new_df['overall'] == 4].sample(4000)
    new_df3 = new_df[new_df['overall'] == 3].sample(8000)
    new_df4 = new_df[new_df['overall'] == 2].sample(4000)
    new_df5 = new_df[new_df['overall'] == 1].sample(4000)
    
    new_df6 = pd.concat([new_df1,new_df2,new_df3,new_df4,new_df5], axis = 0, ignore_index = True)
    
    new_df6.to_csv(str(counter)+".csv", index = False)
    
    new_df = None
    counter = counter + 1


# combining those chunks into a single CSV file i.e balanced_reviews.csv

filenames = glob('*.csv')
dataframes = [pd.read_csv(f, encoding='cp1252') for f in filenames]
frame = pd.concat(dataframes, axis = 0, ignore_index = True)
frame.to_csv('balanced_reviews.csv', index = False)
frame.info()



# reading the dataset and performing the data analysis
df = pd.read_csv(r'balanced_reviews.csv')

df.dropna(inplace = True)

df['overall'] != 3

df = df[df['overall'] != 3]
df

df['overall'].value_counts()

df['Positivity'] = np.where(df['overall'] > 3,1,0)

df['Positivity'].value_counts()

df['reviewText'].head()





# performing the data cleaning with the help of Natural Language tool Kit with the help of stopwords concept
nltk.download('stopwords')



df['reviewText'][0]

df.iloc[0, 1]

corpus = []

for i in range(0, 527357):
    print(i)
    review = re.sub('[^a-zA-Z]', ' ', df.iloc[i, 1])
    
    review = review.lower()
    
    review = review.split()
    
    review = [word for word in review if not word in stopwords.words('english')]
    
    ps =  PorterStemmer()
    
    review = [ps.stem(word) for word in review]
    
    review = " ".join(review)
    
    corpus.append(review)


features = CountVectorizer().fit_transform(corpus)
labels = df.iloc[:,-1]



# bag of words model
features_train, features_test, labels_train, labels_test = train_test_split(df['reviewText'], df['Positivity'], random_state = 42 ) 



vect = CountVectorizer().fit(features_train)
vect.vocabulary_
len(vect.get_feature_names())
features_train_vectorized = vect.transform(features_train)
# features_train_vectorized.toarray()
features_train_vectorized
# vect.get_feature_names()[15000:15005]



# fitting and prediction of model with logistic regression
model = LogisticRegression()
model.fit(features_train_vectorized, labels_train)
predictions = model.predict(vect.transform(features_test))



# use of Tfid Vectorizer as an ideal vectorizer 
vect = TfidfVectorizer(min_df = 5).fit(features_train)
features_train_vectorized = vect.transform(features_train)
# vect
# features_train_vectorized.toarray()

# generation of confusion matrix and estimating the roc accuracy score with the help of the test data and predicted data
confusion_matrix(labels_test, predictions)
roc_auc_score(labels_test, predictions)

# dumping the model into the pickle file and the vectorized vocabulary into a new pickle file i.e feature.pkl 

pkl_filename = "pickle_model.pkl"
with open(pkl_filename, 'wb') as file:
    p.dump(model, file)
p.dump(vect.vocabulary_, open('feature.pkl','wb'))



# predicting and finding the roc accuracy score for that pickle file
with open(pkl_filename, 'rb') as file:
    pickle_model = p.load(file)

pred = pickle_model.predict(vect.transform(features_test))
roc_auc_score(labels_test, pred)

