from korean import chosung_tokenizer
from crawling import connect_api, get_tweets_by_quary
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
import pandas as pd
import re

from transformer import Transformer

def get_tweets():
    quary = "딥러닝"
    api = connect_api()
    tweets = get_tweets_by_quary(api, quary, count = 10000)

    return tweets


def preprocessing(text):
    step1 = text.split("\n")
    step2 = [re.sub("[^ㄱ-ㅎ가-힣 .?!]", "", i) for i in step1]
    step3 = [re.sub(r'\s+', " ", x.strip()) for x in step2]
    step4 = [re.sub(r'[?]+', "?", x.strip()) for x in step3]
    step5 = [re.sub(r'[!]+', "!", x.strip()) for x in step4]
    step6 = []
    for i in step5:
        step6.extend(i.split("!"))
    step7 = []
    for i in step6:
        step7.extend(i.split("?"))
    step8 = []
    for i in step7:
        step8.extend(i.split(". "))
    step8 = list(set(step8))

    return step8


def nlp_tokenize(X, maxlen=100, num_words=1000):
    # 토큰화
    tokenizer = Tokenizer(num_words = num_words)
    tokenizer.fit_on_texts(X)
    X_seq = tokenizer.sequences_to_texts(X)

    # 0 패딩
    X_seq_matrix = sequence.pad_sequences(sequences=X_seq, maxlen=maxlen, padding='post')

    # 워드 인덱싱, 정수 인코딩
    word_encoded = tokenizer.word_index

    return X_seq_matrix, word_encoded


if __name__ == "__main__":
    # string = "오늘 저녁 치킨 고?"
    # print(chosung_tokenizer(string))
    tweets = get_tweets()
    chosungs = []
    labels = []
    for tweet in tweets:
        text = preprocessing(tweet.full_text)
        labels.extend(["\t " + i + " \n" for i in text])
        
    
    labels = list(set(labels))
    chosungs = [chosung_tokenizer(i) for i in labels]

    df = pd.DataFrame(zip(chosungs,labels), columns=["chosung","target"])
    df.to_csv("data.csv")
    print(df)
    
    X_train = df.chosung
    y_train = df.target

