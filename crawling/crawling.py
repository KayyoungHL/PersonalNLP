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


def get_tweets(api, username):
    try:
        tweets = api.home_timeline(username, tweet_mode="extended")
    except:
        tweets = api.home_timeline(username)

    return tweets


if __name__=="__main__":
    username = "KayyoungHL"
    api = connect_api()
    print(get_tweets(api, username)[0].full_text)