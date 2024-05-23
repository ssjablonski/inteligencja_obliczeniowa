import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')



# a) Load the text file
with open('zad1/text.txt', 'r') as file:
    text = file.read().replace('\n', '')

# b) Tokenize the document
tokens = nltk.word_tokenize(text)
print(f"Number of words after tokenization: {len(tokens)}")

# c) Remove stop words
stop_words = set(stopwords.words('english'))
tokens = [word for word in tokens if not word in stop_words]
print(f"Number of words after removing stop words: {len(tokens)}")

# d) Check for any additional unnecessary words and remove them
additional_stop_words = ["'s", ".", ",", '"', "``", "-"]  # Add any additional stop words here
tokens = [word for word in tokens if not word in additional_stop_words]
print(f"Number of words after removing additional stop words: {len(tokens)}")

# e) Lemmatize the document
lemmatizer = WordNetLemmatizer()
tokens = [lemmatizer.lemmatize(word) for word in tokens]
print(f"Number of words after lemmatization: {len(tokens)}")

# f) Create a word vector and plot the 10 most common words
word_vector = Counter(tokens)
most_common_words = word_vector.most_common(10)

words, counts = zip(*most_common_words)
plt.bar(words, counts)
plt.show()

# g) Create a word cloud
wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# Number of words after tokenization: 407
# Number of words after removing stop words: 272
# Number of words after removing additional stop words: 236
# Number of words after lemmatization: 236
