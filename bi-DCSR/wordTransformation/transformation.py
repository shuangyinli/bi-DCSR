from nltk.stem import WordNetLemmatizer

import nltk
from nltk.corpus import words
from nltk.corpus import wordnet

nltk.download('words')
nltk.download('wordnet')

#wordlist = set(words.words())

#print(len(words.words()))



#print(wordnet)

if "distinguished" in words.words():
    print("...")

print(WordNetLemmatizer().lemmatize('goods'))