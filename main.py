from korean import chosung_tokenizer
from crawling import connect_api, get_tweets_by_quary


def get_tweets():
    quary = "딥러닝"
    api = connect_api()
    tweets = get_tweets_by_quary(api, quary, count = 100)

    return tweets


def preprocessing():
    ...


if __name__ == "__main__":
    string = "오늘 저녁 치킨 고?"
    print(chosung_tokenizer(string))
    