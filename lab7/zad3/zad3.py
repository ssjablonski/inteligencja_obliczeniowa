from twikit import Client
import json

# Inicjalizacja klienta Twikit
client = Client('en-US')

# Wczytywanie informacji logowania z pliku JSON
with open("login.json", 'r') as file:
    user_info = json.load(file)

# Logowanie do Twittera
client.login(auth_info_1=user_info["username"], password=user_info["password"])

# Zapisywanie ciasteczek do pliku
client.save_cookies('cookies.json')

# Wczytywanie ciasteczek z pliku
client.load_cookies(path='cookies.json')

# Pobieranie użytkownika według nazwy ekranu
user = client.get_user_by_screen_name("elonmusk")

# Pobieranie tweetów użytkownika
tweets = user.get_tweets('Tweets', count=1000)

# Przygotowanie danych do zapisu
tweets_to_store = []
for tweet in tweets:
    tweets_to_store.append({
        'created_at': tweet.created_at,
        'favorite_count': tweet.favorite_count,
        'full_text': tweet.full_text,
    })

# Zapis tweetów do pliku JSON
with open('tweets.json', 'w', encoding='utf-8') as json_file:
    json.dump(tweets_to_store, json_file, ensure_ascii=False, indent=4)

# Wypisanie tweetów posortowanych według liczby polubień
sorted_tweets = sorted(tweets_to_store, key=lambda x: x['favorite_count'], ascending=False)
print(json.dumps(sorted_tweets, indent=4))
