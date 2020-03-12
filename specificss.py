from IPython import display
import math
from pprint import pprint
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize, RegexpTokenizer
import praw
from nltk.corpus import stopwords
from praw.models import MoreComments


reddit = praw.Reddit(client_id='Kc-VAYjEwxUIew',
                     client_secret='gqt3JE7LmbXOq-e_irmTprc6s_0',
                     user_agent='Bshreyas01')
sns.set(style='darkgrid', context='talk', palette='Dark2')
headlines = set()
com=set()


for submission in reddit.subreddit('coronavirus').hot(limit=10):
    count = 0
    headlines.add(submission.title)
    display.clear_output()
    print(len(headlines))
    print('-----------------------------------------------------------------------------------------------------------')
    for top_level_comment in submission.comments:
        if count == 10:
            break
        if isinstance(top_level_comment, MoreComments):
            continue
        com.add(top_level_comment.body)
        #print(top_level_comment.author)
        count += 1


sia = SIA()
results = []
results1 = []

for line in com:
    pol_score = sia.polarity_scores(line)
    pol_score['comment'] = line
    results.append(pol_score)

pprint(results[:5], width=150)

for line in headlines:
    pol_score = sia.polarity_scores(line)
    pol_score['Headlines'] = line
    results1.append(pol_score)

pprint(results1[:5], width=150)


df = pd.DataFrame.from_records(results)
pprint(df.head(), width=200)

df['label'] = 0
df.loc[df['compound'] > 0.2, 'label'] = 1
df.loc[df['compound'] < 0.2, 'label'] = -1
pprint(df.head(), width=200)

print("Positive headlines:\n")
pprint(list(df[df['label'] == 1].pos)[:5], width=200)
pprint(list(df[df['label'] == 1].comment)[:5], width=200)

print("\nNegative headlines:\n")
pprint(list(df[df['label'] == -1].neg)[:5], width=200)
pprint(list(df[df['label'] == -1].comment)[:5], width=200)

df2 = df[['comment', 'label']]
df2.to_csv('reddit_comments_labels.csv', mode='a', encoding='utf-8', index=True)
df2


