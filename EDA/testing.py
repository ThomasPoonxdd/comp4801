import pandas as pd
import numpy as np

from collections import Counter
import re
def tokenize(text):
    # try:
    text.replace("\n", " ")
    # except:
    #     print(text)
    #     return 0
    N_grams =[]
    words = []
    for t in text.split(" "):
        t = re.sub(r"[^a-zA-Z0-9]", "" , t)
        if len(t) > 0:
            words.append(t)
    # unigram
    freq = Counter(words)
    freq = sorted(freq.items(), key=lambda x: x[1],reverse=True)
    N_grams.append(freq)
    # bigram
    bigram=[pair for pair in zip(words[:-1], words[1:])]
    freq = Counter(bigram)
    freq = sorted(freq.items(), key=lambda x: x[1],reverse=True)
    N_grams.append(freq)
    # trigram
    trigram=[tri for tri in zip(words[:-2], words[1:-1], words[2:])]
    freq = Counter(trigram)
    freq = sorted(freq.items(), key=lambda x: x[1],reverse=True)
    N_grams.append(freq)
    return N_grams

with open("raw_text.txt", 'r') as f:
    df = pd.DataFrame(columns=['text', 'token'])
    for line in f:
        text = line
        token = tokenize(text)
        df = pd.concat([df, pd.DataFrame([{'text': text, 'token': token}])], ignore_index=True)
    df.to_csv('tokenize.csv', encoding='utf-8', index=False)

