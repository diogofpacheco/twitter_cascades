# twitter_cascades
Code to reconstruct Twitter cascades given a set of tweets and a list of followers from the participants in the conversation

The actual code in the `encoding_twitter_cascades` notebook:
 1. read tweets from a mongodb collection.
    1. extract quoted embedded tweets.
    2. extract retweeted embedded tweets.
 2. for each tweet, define its actionType (tweet, retweet, retweet of quote, etc.) and set rootID, parentID, or provisoryParentID accordingly (see `encoding_cascade_functions.checkActionType`).
 3. find best parentID to replace provisoryParent
 4. find parentID for retweets and quotes
 5. find rootID for replies and those initially with provisoryParent
 6. at this point, cascades are reconstructed and can be exported in the Socialsim output format.
 7. compute several metrics/features to generate a cascade summary collection and save to mongo.
