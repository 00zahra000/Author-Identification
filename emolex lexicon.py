from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
import pandas as pd

blob = TextBlob("It would be funny if it weren't so exciting", analyzer=NaiveBayesAnalyzer())
print(blob.sentiment)

blob = TextBlob(
    "Although I don't have a prescription for what others should do, I know I have been very fortunate and feel a responsibility to give back to society in a very significant way.",
    analyzer=NaiveBayesAnalyzer())
print(blob.sentiment)

blob = TextBlob(
    "I believe it's important that we ensure that the police have a modern and flexible workforce. I think that's what is necessary, so that they can provide the public with the service that they want.",
    analyzer=NaiveBayesAnalyzer())
print(blob.sentiment)

filepath = "/home/skyblue/Downloads/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-v0.92/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
emolex_df = pd.read_csv(filepath, names=["word", "emotion", "association"], skiprows=45, sep='\t')
print(emolex_df[15000:].head())
print('\nCycle of Emotions used in EmoLex:\n', emolex_df.emotion.unique())
print('\nNumber of words covered for each Emotion:\n', emolex_df.emotion.value_counts())
print('\nNumber of unique word in each emotion:\n', emolex_df[emolex_df.association == 1].emotion.value_counts())

# Show the words for a unique emotion:
print(emolex_df[(emolex_df.association == 1) & (emolex_df.emotion == 'joy')].word)

emolex_words = emolex_df.pivot(index='word', columns='emotion', values='association').reset_index()
print(emolex_words.head())

# Emotions of a certain word
print(emolex_words[emolex_words.word == 'loyal'])
print('Emotion of Politics:')
print(emolex_words[emolex_words.word == 'politics'])

# All the words with a certain emotion
print(emolex_words[emolex_words.surprise == 1].head())
