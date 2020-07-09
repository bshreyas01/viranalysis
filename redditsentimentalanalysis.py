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


reddit = praw.Reddit(client_id='Kc-VAYjEwxUIew',
                     client_secret='',
                     user_agent='')
sns.set(style='darkgrid', context='talk', palette='Dark2')
headlines = set()
for submission in reddit.subreddit('coronavirus').hot(limit=None):
    headlines.add(submission.title)
    display.clear_output()
    print(len(headlines))

sia = SIA()
results = []

for line in headlines:
    pol_score = sia.polarity_scores(line)
    pol_score['headline'] = line
    results.append(pol_score)

pprint(results[:5], width=150)

df = pd.DataFrame.from_records(results)
pprint(df.head(), width=200)

df['label'] = 0
df.loc[df['compound'] > 0.2, 'label'] = 1
df.loc[df['compound'] < -0.2, 'label'] = -1
pprint(df.head(), width=200)

print("Positive headlines:\n")
pprint(list(df[df['label'] == 1].pos)[:5], width=200)
pprint(list(df[df['label'] == 1].headline)[:5], width=200)

print("\nNegative headlines:\n")
pprint(list(df[df['label'] == -1].neg)[:5], width=200)
pprint(list(df[df['label'] == -1].headline)[:5], width=200)

df2 = df[['headline', 'label']]
df2.to_csv('reddit_headlines_labels.csv', mode='a', encoding='utf-8', index=True)

print(df.label.value_counts())
counts = df.label.value_counts(normalize=True) * 100
print(counts)
fig, ax = plt.subplots(figsize=(8, 8))

ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
ax.set_ylabel("Percentage")

sns.barplot(x=counts.index, y=counts, ax=ax)
plt.show()

example = "This is an example sentence! However, it isn't a very informative one"

#print(word_tokenize(example, language='english'))

tokenizer = RegexpTokenizer(r'\w+')
#print(tokenizer.tokenize(example))


stop_words = stopwords.words('english')


def process_text(headlines):
    tokens1 = []
    for line in headlines:
        toks = tokenizer.tokenize(line)
        toks = [t.lower() for t in toks if t.lower() not in stop_words]
        tokens1.extend(toks)

    return tokens1


pos_lines = list(df[df.label == 1].headline)

pos_tokens = process_text(pos_lines)
print(pos_tokens)
pos_freq = nltk.FreqDist(pos_tokens)

print(pos_freq.most_common(20))

y_val = [x[1] for x in pos_freq.most_common()]

fig = plt.figure(figsize=(10,5))
plt.plot(y_val)

plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Word Frequency Distribution (Positive)")#
plt.show()

y_final = []
for i, k, z, t in zip(y_val[0::4], y_val[1::4], y_val[2::4], y_val[3::4]):
    y_final.append(math.log(i + k + z + t))

x_val = [math.log(i + 1) for i in range(len(y_final))]

fig = plt.figure(figsize=(10,5))

plt.xlabel("Words (Log)")
plt.ylabel("Frequency (Log)")
plt.title("Word Frequency Distribution (Positive)")
plt.plot(x_val, y_final)
plt.show()

neg_lines = list(df2[df2.label == -1].headline)

neg_tokens = process_text(neg_lines)
neg_freq = nltk.FreqDist(neg_tokens)

neg_freq.most_common(20)



y_val = [x[1] for x in neg_freq.most_common()]

fig = plt.figure(figsize=(10,5))
plt.plot(y_val)

plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Word Frequency Distribution (Negative)")
plt.show()

y_final = []
for i, k, z in zip(y_val[0::3], y_val[1::3], y_val[2::3]):
    if i + k + z == 0:
        break
    y_final.append(math.log(i + k + z))

x_val = [math.log(i+1) for i in range(len(y_final))]

fig = plt.figure(figsize=(10,5))

plt.xlabel("Words (Log)")
plt.ylabel("Frequency (Log)")
plt.title("Word Frequency Distribution (Negative)")
plt.plot(x_val, y_final)
plt.show()



