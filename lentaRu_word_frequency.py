import re
import nltk
import requests
import pandas as pd
from bs4 import BeautifulSoup
from pymystem3 import Mystem


def get_news_urls(rubric_url, num_of_news=10):
    response = requests.get(rubric_url).text
    soup = BeautifulSoup(response, 'html.parser')
    urls = ['https://lenta.ru' + url.get('href') for url in soup.find_all('a')
            if 'news' in str(url) and 'lenta.ru' not in str(url)]
    if len(urls) < num_of_news:
        return urls
    return urls[:num_of_news]


def get_text(url_list):
    all_in_one_text = ''
    for url in url_list:
        response = requests.get(url).text
        soup = BeautifulSoup(response, 'html.parser')
        news_texts = [tag.get_text() for tag in soup.find_all({'p'})]
        all_in_one_text += ' '.join(news_texts)
    return all_in_one_text


def preprocess_text(text, stop_words):
    mystem = Mystem()
    punctuation_garbage = r'\w?[\s\d\.,\-—_=\+/!”;:%\?\*\(\)\[\]«»><]*'
    lemmas = mystem.lemmatize(text.lower())
    lemmas = [lemma for lemma in lemmas if lemma.strip() not in stop_words
              and lemma not in re.findall(punctuation_garbage, lemma)]
    return lemmas


def calculate_frequency(lemmas, top=20):
    frequency = nltk.FreqDist(lemmas).most_common(top)
    return frequency


def save_to_csv(frequency, rubric):
    df = pd.DataFrame(frequency, columns=["Слово", "Частота"])
    df.to_csv('{}.csv'.format(rubric.split('/')[-1]), index=None, header=True)
    print('Создаем csv для рубрики: {}'.format(rubric))


with open('rubrics.txt') as reader:
    rubrics = reader.read().split('\n')

with open('stopwords.txt') as reader:
    stopwords = reader.read().split('\n')

for rubric_name in rubrics:
    news_urls = get_news_urls(rubric_name, 100)
    plain_text = get_text(news_urls)
    text_lemmas = preprocess_text(plain_text, stopwords)
    lemmas_frequency = calculate_frequency(text_lemmas)
    save_to_csv(lemmas_frequency, rubric_name)

print('--end')
