# Imported
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re
import nltk
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import chardet
import streamlit as st

# Reading Dataset
fh = '/home/samarth/Documents/python codes/Ml basics/Projects/NLP-Bag Of Words/mail.csv'
with open(fh, 'rb') as mail_raw:
    coding = chardet.detect(mail_raw.read(100000))
dataset = pd.read_csv('mail.csv', encoding='Windows-1252')
dataset.drop('S. No.', axis='columns', inplace=True)
dataset.head()

# Encoding Categorical Variable
le = LabelEncoder()
dataset['Label'] = le.fit_transform(dataset['Label'])

# Processing and Simplifying Text Entries Using Stemming and Regular Expressions
nltk.download('stopwords')
corpus = []
for i in range(0, 957):
    mail = re.sub('[^a-zA-Z]', ' ', dataset['Message_body'][i])
    mail = mail.lower()
    mail = mail.split()
    ps = PorterStemmer()
    stpwords = stopwords.words('english')
    stpwords.remove('not')
    mail = [ps.stem(word) for word in mail if word not in set(stpwords)]
    mail = ' '.join(mail)
    corpus.append(mail)

# Converting Processed Text Into Array (Bag Of Words)
cv = CountVectorizer(max_features=2360)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values

# Splitting Dataset Into Test and Train Batch
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=0)

# Training Model on the Train Data
classifier = GaussianNB()
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

# Confusion Matrix and Accuracy
cm = confusion_matrix(y_pred, y_test)
print(cm)
accuracy = accuracy_score(y_pred, y_test)
print(accuracy)

# Classification Based On Single Input And Streamlit

st.title("SPAM CLASSIFIER")
st.subheader("Enter the email : ")


def Single_input_output():
    mail_inp = st.text_area("Input")
    mail_inp = re.sub('[^a-zA-Z]', ' ', mail_inp)
    mail_inp = mail_inp.lower()
    mail_inp = mail_inp.split()
    mail_inp = [ps.stem(inp) for inp in mail_inp if inp not in set(stpwords)]
    mail_inp = ' '.join(mail_inp)
    mail_inp = [mail_inp]
    mail_inp = cv.transform(mail_inp).toarray()
    print(mail_inp)
    y_pred = classifier.predict(mail_inp)
    return y_pred


prediction = Single_input_output()

st.subheader("Result :")
if prediction == 0:
    st.write('Not Spam')
else:
    st.write('Spam')

st.sidebar.markdown('### Spam Examples')
st.sidebar.write(
    "1.PRIVATE! Your 2003 Account Statement for shows 800 un-redeemed S.I.M. points. Call 08718738001 Identifier Code: 49557 Expires 26/11/04")
st.sidebar.write("2.URGENT We are trying to contact you Last weekends draw shows u have won a £1000 prize GUARANTEED Call 09064017295 Claim code K52 Valid 12hrs 150p pm")
