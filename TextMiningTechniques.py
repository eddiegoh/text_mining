# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 20:28:30 2018

@author: Eddie
For couple customers what appeared to be main attraction point and how can we
further develop such customers by providing additional services around the
main attraction point?
"""
import pycountry
import pandas as pd
from nltk.corpus import stopwords
from textblob import Word
import nltk
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures

review = pd.read_csv('TripAdvisorReviewsCouple.csv')  # create data frame from data extracted from trip advisor

print(review.info())  # review the detail of data frame

# The rating was an image on the website so what was scrap was the class name hence formatting was required
# format rating column to integer
review["rate"] = review["rating"].str.replace("[bubble_]", '')
review["rate"] = review["rate"].str.replace("[0]", '')

# The nationality seem to be in free text form hence some of the value is the state of the country
# We tried using Pycountry to change the state name to its country name
# Result wasn't seem to very good
# Note that when you run it will keep prompting error list index out of range
countries_list = review.loc[:, 'country']  # create a series containing country data only
country_name = []  # create a list to store the cleaned country data

for i, name in enumerate(countries_list):
    if name == 'N.A.':
        country_name.append('N.A.')
    else:
        for country in pycountry.countries:
            if country.name.lower() in name.lower():  # converted to lowercase to improve the cleaning result
                country_name.append(country.name)
        try:  # this try and except are for those rows where country data is present but not found in pycountry
            if country_name[i] == '':
                print('catching exception')
        except Exception as error:
            print(error)
            country_name.append('N.A')

# joining back the cleaned country data back to the data frame in a new column
se = pd.Series(country_name)
review['Country Name'] = se.values

# count the number of words in a feedback
review["word_count"] = review["review"].apply(lambda x: len(str(x).split(" ")))

# count the number of time a digit appear
review["numeric_count"] = review["review"].apply(lambda x: len([x for x in x.split() if x.isdigit()]))

# Display the first original comment
print(review['review'][0])

# convert word to lower case and stored in a new column processed_feedback
review["processed_feedback"] = review["review"].apply(lambda x: " ".join(x.lower() for x in x.split()))

# Display the processed comment to verify changes
print(review['processed_feedback'][0])

# remove n't replace with not e.g. didn't to did not
# I wanted to keep the 'not' word so that when removing stop word it wont change the meaning of the content
review["processed_feedback1"] = review["processed_feedback"].str.replace("[n]'[t]", ' not ')
review["processed_feedback1"] = review["processed_feedback1"].str.replace('[^\w\s]', ' ')  # remove punctuation
review["processed_feedback1"] = review["processed_feedback1"].str.replace('\s+', ' ')  # remove extra spacing
print(review['processed_feedback1'][0])

# Removing of stop words
# assign stop words to stop
stop = stopwords.words('english')
stop.remove('no')
stop.remove('nor')
stop.remove('not')
review["processed_feedback2"] = review["processed_feedback1"].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
print(review['processed_feedback2'][0])


# Performing Lemmatization
review["processed_feedback3"] = review["processed_feedback2"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
# Display the processed comment to verify changes
print(review['processed_feedback3'][0])

# view the frequency of each word being used
freq_words = pd.Series(' '.join(review["processed_feedback3"]).split()).value_counts()[:]
freq_words.tolist()

# view the number of words that appeared only once
count = 0
for x in freq_words:
    if x == 1:
        count += 1
print('%d words appeared once' % count)

# view the least frequently appeared words
least_freq_words = pd.Series(' '.join(review["processed_feedback3"]).split()).value_counts()[-7530:]
least_freq_words = list(least_freq_words.index)

# view the most frequently appeared words
most_freq_words = pd.Series(' '.join(review["processed_feedback3"]).split()).value_counts()[0:13]
most_freq_words = list(most_freq_words.index)


# Perform POS tagging to extract proper nouns and common nouns
# This was done only done on a small data set.
# Need to refactor the code so that it can process large data set
tokenized_sent = []
tagged_sent = []
review_10 = review.loc[:10, 'processed_feedback3']

for comment in review_10:
    tokenized_sent.append(nltk.word_tokenize(comment))
    for words in tokenized_sent:
        tagged_sent.append(nltk.pos_tag(words))

print(tagged_sent[:2])
Noun_list = []
for tagged_words_list in tagged_sent:
    for word in tagged_words_list:
        if word[1] == 'NN' or word[1] == 'NNP':
            Noun_list.append(word[0])
print(Noun_list)


# Create Word Cloud for data visualization
text = " ".join(review for review in review.processed_feedback3)
print("There are {} words in the combination of all review.".format(len(text)))
# Generate a word cloud image
word_cloud = WordCloud(background_color="white").generate(text)

# Display the generated image in the matplotlib way:
plt.imshow(word_cloud, interpolation='bilinear')
# remove the axis and save the word cloud
plt.axis("off")
plt.show()
word_cloud.to_file("wordcloud.png")

# Create Bigram and Trigram from review
tokenized_text = nltk.word_tokenize(text)
bcf = BigramCollocationFinder.from_words(tokenized_text)
# There are different metrics so we can try different metric to see the result
top20 = bcf.nbest(BigramAssocMeasures.likelihood_ratio, 20)
# top20 = bcf.nbest(BigramAssocMeasures.raw_freq, 20)
print(top20)
tcf = TrigramCollocationFinder.from_words(tokenized_text)
tcf_top20 = tcf.nbest(TrigramAssocMeasures.likelihood_ratio, 20)
print(tcf_top20)






