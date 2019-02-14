"""
PNNL code for cascade reconstruction - https://github.com/pnnl/socialsim/blob/master/december-measurements/cascade_reconstruction/twitter_cascade_reconstruction.py

copied on 01/29/19, master branch, version 67ca025.
"""
import pandas as pd
import glob
from collections import defaultdict
import os
import pprint
import json
import numpy as np

def load_data(json_file, full_submission=True):
    """
    Takes in the location of a json file and loads it as a pandas dataframe.
    Does some preprocessing to change text from unicode to ascii.
    """

    if full_submission:
        with open(json_file) as f:
            dataset = json.loads(f.read())

        dataset = dataset['data']
        dataset = pd.DataFrame(dataset)
    else:
        dataset = pd.read_json(json_file)

    dataset.sort_index(axis=1, inplace=True)
    dataset = dataset.replace('', np.nan)

    # This converts the column names to ascii
    mapping = {name:str(name) for name in dataset.columns.tolist()}
    dataset = dataset.rename(index=str, columns=mapping)

    # This converts the row names to ascii
    dataset = dataset.reset_index(drop=True)

    # This converts the cell values to ascii
    json_df = dataset.applymap(str)

    return dataset


class ParentIDApproximation:
    """
    class to obtain parent tweet id for retweets
    """

    def __init__(self, followers, cascade_collection_df, nodeID_col_name="nodeID", userID_col_name='nodeUserID',
                 nodeTime_col_name='nodeTime', rootID_col_name='rootID',
                 root_userID_col_name='rootUserID',
                 root_nodeTime_col_name='rootTime'):
        """
        :param followers: dictionary with key: userID, value: [list of followers of userID]
        :param cascade_collection_df: dataframe with nodeID, userID, nodeTime, rootID, root_userID, root_nodeTime as columns
                default values for column names correspond to those in the Twitter data schema
                (https://wiki.socialsim.info/display/SOC/Twitter+Data+Schema)
        """
        self.followers = followers
        self.cascade_collection_df = cascade_collection_df
        self.nodeID_col_name = nodeID_col_name
        self.userID_col_name = userID_col_name
        self.nodeTime_col_name = nodeTime_col_name
        self.rootID_col_name = rootID_col_name
        self.root_userID_col_name = root_userID_col_name
        self.root_nodeTime_col_name = root_nodeTime_col_name

    def get_all_tweets_rtd_later_by_followers(self, tweet_id, cascade_df):
        tweet_details = cascade_df.loc[tweet_id]
        
        # add self to followers because users will retweet themselves
        return cascade_df[
            (cascade_df[self.userID_col_name].
                isin(self.followers.get(tweet_details[self.userID_col_name],set()).union(
                {tweet_details[self.userID_col_name]}))) &  # in followers
            (cascade_df[self.nodeTime_col_name] > tweet_details[self.nodeTime_col_name])
            ]. \
            index.values.tolist()

    def update_parentid(self, cascade_df_main, root_id):
        root_userID = cascade_df_main.loc[cascade_df_main.index.max()][self.root_userID_col_name]
        root_nodeTime = cascade_df_main.loc[cascade_df_main.index.max()][self.root_nodeTime_col_name]
        cascade_df = cascade_df_main.sort_values(self.nodeTime_col_name).drop(
            [self.root_userID_col_name, self.root_nodeTime_col_name], axis=1).copy()
        cascade_df["parentID"] = 0
        # root tweet also added to the cascade since we need the time when the root tweet was tweeted
        cascade_df = cascade_df.append({
            self.nodeID_col_name: root_id,
            self.userID_col_name: root_userID,
            self.nodeTime_col_name: root_nodeTime,
            self.rootID_col_name: root_id,
            "parentID": None
        },ignore_index=True)
#         cascade_df.loc[cascade_df.index.max() + 1] = {
#             self.nodeID_col_name: root_id,
#             self.userID_col_name: root_userID,
#             self.nodeTime_col_name: root_nodeTime,
#             self.rootID_col_name: root_id,
#             "parentID": None
#         }
        cascade_df = cascade_df.set_index(self.nodeID_col_name)
        seed_tweets = [root_id]
        while seed_tweets:
            new_seed_tweets = []
            for seed_tweet_id in seed_tweets:
                tweets_to_be_updated = self.get_all_tweets_rtd_later_by_followers(seed_tweet_id,
                                                                                  cascade_df)  # assume a user as their follower since a user can retweet themselves
                cascade_df.loc[tweets_to_be_updated, "parentID"] = seed_tweet_id
                new_seed_tweets.extend(tweets_to_be_updated)

            seed_tweets = cascade_df[
                cascade_df.index.isin(new_seed_tweets)].index.tolist()  # keeping the order a.t. tweeted timestamp
        cascade_df.dropna(subset=["parentID"])
        return cascade_df[cascade_df["parentID"] != 0].reset_index()

    def get_approximate_parentids(self, mapping_only=True, csv=False):
        """
        :param mapping_only: remove other columns except nodeID and parentID
        :param csv: write the parentID mapping to a csv file
        """
        # parentID is None for root tweets
        parentid_map_dfs = []
        for tweet_id, cascade_df in self.cascade_collection_df.groupby(self.rootID_col_name):
            updated_cascade_df = self.update_parentid(cascade_df, tweet_id)
            parentid_map_dfs.append(updated_cascade_df)
        parentid_map_all_cascades_df = pd.concat(parentid_map_dfs).reset_index(drop=True)
        parentid_map_all_cascades_df.dropna(inplace=True)
        if mapping_only:
            parentid_map_all_cascades_df = parentid_map_all_cascades_df[[self.nodeID_col_name, "parentID"]]
        if csv:
            parentid_map_all_cascades_df.to_csv("retweet_cascades_with_parentID.csv", index=False)

        return parentid_map_all_cascades_df

def get_reply_cascade_root_tweet(df, parent_node_col="parentID", node_col="nodeID", root_node_col="rootID", timestamp_col="nodeTime", json=False):
    """
    :param df: dataframe containing a set of reply cascades
    :param json: return in json format or pandas dataframe
    :return: df with rootID column added, representing the cascade root node
    """
    df = df.sort_values(timestamp_col)
    rootid_mapping = pd.Series(df[parent_node_col].values, index=df[node_col]).to_dict()

    def update_reply_cascade(reply_cascade):
        for tweet_id, reply_to_tweet_id in reply_cascade.items():
            if reply_to_tweet_id in reply_cascade:
                reply_cascade[tweet_id] = reply_cascade[reply_to_tweet_id]
        return reply_cascade

    prev_rootid_mapping = {}
    while rootid_mapping != prev_rootid_mapping:
        prev_rootid_mapping = rootid_mapping.copy()
        rootid_mapping = update_reply_cascade(rootid_mapping)
    df["rootID_new"] = df[node_col].map(rootid_mapping)
#     df.loc[df['actionType'] == 'reply','rootID'] =     df.loc[df['actionType'] == 'reply','rootID_new']
    df.loc[~df['actionType'].isin(['retweet', 'tweet', 'quote']),'rootID'] =     df.loc[~df['actionType'].isin(['retweet', 'tweet', 'quote']),'rootID_new']
    df = df.drop('rootID_new',axis=1)
    if json:
        return df.to_json(orient='records')
    else:
        return df


if __name__ == '__main__':

    #one text file per user listing that user's followers
    follower_data = glob.glob('example_follower_data/*.txt')

    #create followers dictionary with user IDs as keys and list of followers as values    
    followers = defaultdict(lambda: set([]))
    for fn in follower_data:
        user = os.path.splitext(os.path.split(fn)[-1])[0]
        f = set(pd.read_csv(fn,header=None)[0].tolist())
        print('User {}: {} followers'.format(user,len(f)))
        followers[user] = f

    #read in ground truth data file in JSON format
    #this data should be missing parentIDs for retweets/quotes and rootIDs for replies 
    #(because they are not available from the Twitter JSON) 
    cascade_collection_df = load_data('twitter_reconstruction_example_data.json',full_submission=False)

    #store replies for later
    replies = cascade_collection_df[cascade_collection_df['actionType'] == 'reply']

    #limit data to events where the rootID is also contained in the data
    cascade_collection_df = cascade_collection_df[cascade_collection_df['rootID'].isin(cascade_collection_df['nodeID'])]

    #get the user who posted the root tweet for each retweet
    root_users = cascade_collection_df[['nodeID','nodeUserID','nodeTime']]
    root_users.columns = ['rootID','rootUserID','rootTime']
    cascade_collection_df = cascade_collection_df.merge(root_users,on='rootID',how='left')

    #store original tweets for later
    original_tweets = cascade_collection_df[cascade_collection_df['actionType'] == 'tweet']

    #subset on only retweets and quotes
    cascade_collection_df = cascade_collection_df[cascade_collection_df['actionType'].isin(['retweet','quote'])]
    cascade_collection_df_retweets = cascade_collection_df[['nodeID','nodeUserID','nodeTime','rootID','rootUserID','rootTime']]

    #get parent IDs for retweets and quotes
    pia = ParentIDApproximation(followers, cascade_collection_df_retweets)
    parent_ids = pia.get_approximate_parentids()

    cascade_collection_df['parentID'] = cascade_collection_df['nodeID'].map(dict(zip(parent_ids.nodeID,parent_ids.parentID)))

    #rejoin with replies and original tweets
    cascade_collection_df = pd.concat([cascade_collection_df,replies,original_tweets],axis=0).sort_values('nodeTime')
    cascade_collection_df = cascade_collection_df.drop(['rootUserID','rootTime'],axis=1)

    #follow cascade chain to get root node for reply tweets
    cascade_collection_df = get_reply_cascade_root_tweet(cascade_collection_df)
    
    print('Results:')
    print(cascade_collection_df)

    output = cascade_collection_df.to_dict(orient='records')

    with open('twitter_example_data_reconstructed.json','w') as f:
        json.dump(output, f)
