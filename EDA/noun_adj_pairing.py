import spacy
import pandas as pd

nlp = spacy.load('en_core_web_sm')

with open('lon_raw_text.txt') as f:
    for line in f:
        line = line.lower()
        doc = nlp(line)

        noun_adj_pairs = {}
        for chunk in doc.noun_chunks:
            adj = []
            noun = ""
            for tok in chunk:
                if tok.pos_ == "NOUN":
                    noun = tok.text
                if tok.pos_ == "ADJ":
                    adj.append(tok.text)
            if noun:
                noun_adj_pairs.update({noun:adj})

        with open('pairs.txt', 'a') as f_2:
            txt = str(noun_adj_pairs)
            if len(txt) > 2:
                f_2.write(txt)
                f_2.write('\n')
