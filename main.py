from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import pandas as pd

df = pd.read_csv('/home/skyblue/Quotes_train_2.csv')
print(df.head())

col = ['name', 'sentence']
df = df[col]
df = df[pd.notnull(df['sentence'])]
df.columns = ['name', 'sentence']
df['author_id'] = df['name'].factorize()[0]
author_id_df = df[['name', 'author_id']].drop_duplicates().sort_values('author_id')
author_to_id = dict(author_id_df.values)
id_to_author = dict(author_id_df[['author_id', 'name']].values)
print(df.head(100))

fig = plt.figure(figsize=(8, 6))
df.groupby('name').sentence.count().plot.bar(ylim=0)
plt.show()

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2),
                        stop_words='english')

features = tfidf.fit_transform(df.sentence).toarray()
labels = df.author_id
print(features.shape)

N = 2
for name, author_id in sorted(author_to_id.items()):
    features_chi2 = chi2(features, labels == author_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("*** Name: {}".format(name))
    print("* Most Correlated Unigrams\n{}".format('\n'.join(unigrams[-N:])))
    print("* Most Correlated Bigrams\n{}".format('\n'.join(bigrams[-N:])))

models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB()]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])

sns.boxplot(x='model_name', y='accuracy', data=cv_df)
sns.stripplot(x='model_name', y='accuracy', data=cv_df,
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()
accuracy_grouping = cv_df.groupby('model_name').accuracy.mean()
print(accuracy_grouping)

model = LinearSVC()
X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index,
                                                                                 test_size=0.30, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

conf_mat = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=author_id_df.name.values, yticklabels=author_id_df.name.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

print(metrics.classification_report(y_test, y_pred, target_names=df['name'].unique()))

## TRAINING THE MODEL
X_train, X_test, y_train, y_test = train_test_split(df['sentence'], df['name'], random_state=0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)

####### SENTIMENT ANALYSIS : EMOLEX EMOTIONAL LEXICON

filepath = "/home/skyblue/Downloads/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
emolex_df = pd.read_csv(filepath, names=["word", "emotion", "association"], skiprows=45, sep='\t')
# print(emolex_df[15000:].head())
# print('\nCycle of Emotions used in EmoLex:\n', emolex_df.emotion.unique())

# print('\nNumber of words covered for each Emotion:\n', emolex_df.emotion.value_counts())
# print('\nNumber of unique word in each emotion:\n', emolex_df[emolex_df.association == 1].emotion.value_counts())

# Show the words for a unique emotion:
# print(emolex_df[(emolex_df.association == 1) & (emolex_df.emotion == 'joy')].word)

emolex_words = emolex_df.pivot(index='word', columns='emotion', values='association').reset_index()
# print(emolex_words.head())

# Emotions of a certain word
# print(emolex_words[emolex_words.word == 'loyal'])

# All the words with a certain emotion
# print(emolex_words[emolex_words.surprise == 1].head())
q = [input(print('Enter a Quote, please:'))]
print('*** Predicted Author ****')
print('\t\t\t', clf.predict(count_vect.transform(q)))
# Some samples from the dataset for the above
# "I guess when people ask what is the biggest transition to the NBA from college, it is definitely defense and the mental part."
# Expectations are a form of first-class truth: If people believe it, it's true.
# John Kerry believes in an America where hard work is rewarded.
print('*** Predicted Sentiment ***')
blob = TextBlob(str(q), analyzer=NaiveBayesAnalyzer())
print('\t\t\t', blob.sentiment)
