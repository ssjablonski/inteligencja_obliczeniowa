import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import text2emotion as te

nltk.download('vader_lexicon')

# a) Positive and negative reviews
with open('nice.txt', 'r') as file:
    positive_review = file.readline().strip()

with open('bad.txt', 'r') as file:
    negative_review = file.readline().strip()
# b) Use Vader to analyze sentiment
sia = SentimentIntensityAnalyzer()

positive_scores = sia.polarity_scores(positive_review)
negative_scores = sia.polarity_scores(negative_review)

print(f"Positive review scores: {positive_scores}")
print(f"Negative review scores: {negative_scores}")

# Positive review scores: {'neg': 0.0, 'neu': 0.67, 'pos': 0.33, 'compound': 0.979}
# Negative review scores: {'neg': 0.294, 'neu': 0.706, 'pos': 0.0, 'compound': -0.9783}

# c) Use pattern to analyze emotions
from pattern.en import sentiment

positive_sentiment = sentiment(positive_review)
negative_sentiment = sentiment(negative_review)

print(f"Positive review sentiment: {positive_sentiment}")
print(f"Negative review sentiment: {negative_sentiment}")

# Positive review sentiment: (0.5045454545454545, 0.5977272727272728)
# Negative review sentiment: (-0.6222222222222222, 0.811111111111111)
# pierwsza wartosc od -1 do 1 (im blizej -1 to negatywna, 0 neutralna, 1 pozytywna), druga od 0 do 1 (im blizej 0 tym bardziej obiektywna, blizej 1 to subiektywna)


#  uwazam ze wyniki testu są trafne i zgodne z przekazanymi emocjami w recenzjach