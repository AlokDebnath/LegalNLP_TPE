from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer
import os
# ps = PorterStemmer()

f = open('sample_text.txt', 'r')
text = f.read()

stopWords = set(stopwords.words("english"))
words = word_tokenize(text)

freqTable = dict()
for word in words:
    word = word.lower()
    if word in stopWords:
        continue
    if word in freqTable:
        freqTable[word] += 1
    else:
        freqTable[word] = 1

sentences = sent_tokenize(text)
#print(sentences)
sentenceValue = dict()
for sentence in sentences:
    for wordValue in freqTable:
        if wordValue in sentence.lower():
            if sentence[:12] in sentenceValue:
                sentenceValue[sentence[:12]] += freqTable[wordValue]
            else:
                sentenceValue[sentence[:12]] = freqTable[wordValue]
sumValues = 0
#print (sentenceValue)

for sentence in sentenceValue:
    #print (sentenceValue[sentence])
    sumValues += sentenceValue[sentence]

# Average value of a sentence from original text
average = int(sumValues/ len(sentenceValue))
summary = ''

for sentence in sentences:
    if sentence[:12] in sentenceValue and sentenceValue[sentence[:12]] > (1.5*average):
        summary += sentence + " "

print (summary)