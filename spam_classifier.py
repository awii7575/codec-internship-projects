# spam email classifier
# using naive bayes to detect spam messages

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# load the dataset
# download from: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
# save it as 'spam.csv' in the same folder

df = pd.read_csv('spam.csv', encoding='latin-1')

# keep only the useful columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

print("first 5 rows:")
print(df.head())
print()
print("spam vs not spam count:")
print(df['label'].value_counts())

# convert labels to numbers  ham=0, spam=1
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

# split into features and target
x = df['message']
y = df['label_num']

# train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# convert text to numbers using bag of words
cv = CountVectorizer()
x_train_cv = cv.fit_transform(x_train)
x_test_cv = cv.transform(x_test)

# train the model
model = MultinomialNB()
model.fit(x_train_cv, y_train)

# predict
y_pred = model.predict(x_test_cv)

# check accuracy
acc = accuracy_score(y_test, y_pred)
print(f"\nmodel accuracy: {acc * 100:.2f}%")

# confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()
print("confusion matrix saved as confusion_matrix.png")

# bar chart - spam vs ham
plt.figure(figsize=(5, 4))
df['label'].value_counts().plot(kind='bar', color=['steelblue', 'tomato'])
plt.title('Spam vs Ham Count')
plt.xlabel('Label')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('spam_ham_count.png')
plt.show()
print("bar chart saved as spam_ham_count.png")

# test with custom messages
def check_message(msg):
    msg_cv = cv.transform([msg])
    result = model.predict(msg_cv)[0]
    if result == 1:
        print(f'"{msg}" --> SPAM')
    else:
        print(f'"{msg}" --> NOT SPAM')

print("\ntesting some messages:")
check_message("Congratulations! You won a free iPhone. Click here to claim now!")
check_message("Hey, are we still meeting tomorrow at 5?")
check_message("URGENT: Your bank account has been suspended. Call now.")
check_message("Can you send me the notes from today's class?")
