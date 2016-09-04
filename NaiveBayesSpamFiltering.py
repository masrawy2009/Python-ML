

spam_data = pd.read_csv("sms_spam_short.csv")
spam_data.dtypes
spam_data.describe()
spam_data.head()

#Text Preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = spam_data.text
vectorizer = TfidfVectorizer(stop_words='english')
tfidf=vectorizer.fit_transform(corpus).todense()
tfidf.shape
tfidf

"""
Modeling and Prediction
"""
#Split into training and testing sets

predictors = tfidf
targets = spam_data.type

pred_train, pred_test, tar_train, tar_test  =   train_test_split(predictors, targets, test_size=.3)

pred_train.shape
pred_test.shape
tar_train.shape
tar_test.shape

#Build model on training data
from sklearn.naive_bayes import GaussianNB

classifier=GaussianNB()
classifier=classifier.fit(pred_train,tar_train)

predictions=classifier.predict(pred_test)

sklearn.metrics.confusion_matrix(tar_test,predictions)
sklearn.metrics.accuracy_score(tar_test, predictions)
sklearn.metrics.classification_report(tar_test, predictions)

