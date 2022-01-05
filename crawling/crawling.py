import tweepy

def connect_api():
    api_key = 'uOnLtambY2KXJL95HGorqtKeX'
    api_key_secret = 'XHyskJ8RSDnYLqn6PXGn40HfrhRPSrt0j2TXKv0oTQp581Ek77'
    bearer_token = 'AAAAAAAAAAAAAAAAAAAAAEPGWAEAAAAAqFFB0w%2FK7HxPJNB7R3ogzSkOrJU%3D0eenKdxINkMzncCNXBijyZf8HZYR9QAI3bxnj9fNEx397BJuof'
    access_token = '1463862729085755396-4cDvkmXJSVRDJp5yKNjReKefWt8mQc'
    access_token_secret = 'uDTff2uBI7a13IXM3Bdu3KM9Bl0BfLWHEbPoa3DeHdsPe'

    auth = tweepy.OAuthHandler(api_key, api_key_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    return api


def get_tweets_by_username(api, username):
    try:
        tweets = api.user_timeline(user_id = username, tweet_mode="extended")
    except:
        tweets = api.user_timeline(user_id = username)

    return tweets

def get_tweets_by_quary(api, quary, count):
    try:
        tweets = api.search_tweets(quary, tweet_mode = "extended", lang = "ko", count = count)
    except:
        tweets = api.search_tweets(quary, lang = "ko", count = count)

    return tweets


if __name__=="__main__":
    username = "KayyoungHL"
    api = connect_api()
    print(get_tweets_by_username(api, username)[0].full_text)
    # quary = "딥러닝"
    # i = 1
    # tweets = api.search_tweets(quary, tweet_mode = "extended", lang = "ko", count = 100)
    
    # for i in tweets:
    #     print(i.full_text)