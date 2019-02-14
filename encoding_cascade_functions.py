import networkx as nx
import numpy as np
import pandas as pd


def extractFeatures(cascade_data):
    """extract the features of a given cascade tree.

    input
    -----
    An object with 'edges' and 'nodes' as keys. For instance, it can be
    a row of a dataframe that has 'nodes' and 'edges' columns. The value for
    the 'edges' should be a temporally ordered edge list, where the root of the
    tree is the destination of the first edge.

    """
    try:
        features = {}
        edges_list = cascade_data["edges"]
        G = nx.DiGraph(edges_list)
        if G:
            first_edge = edges_list[0]
            root_id = first_edge[1]
            features.update(calculateNetworkMetrics(G, root_id, cascade_data["nodes"]))
        return pd.Series(features)
    except nx.NetworkXNoPath as e:
        logging.error("Error extracting feature from: {}".format(cascade_data))
        logging.error(e)


def calculateNetworkMetrics(G, root, nodes):
    short_paths_from_root = [nx.shortest_path_length(G, s, root) for s in nodes]
    # np.unique returns 2 arrays, the keys counted and their counts.
    # the keys will be the distances from the root, i.e. from 0 to max depth,
    # while the counts are the breadth per depth.
    depth, breadth = np.unique(list(short_paths_from_root), return_counts=True)
    return {
        "depth": [
            max(short_paths_from_root[: (idx + 1)])
            for idx in range(len(short_paths_from_root))
        ],
        "depth_max": max(depth),
        "breadth": breadth.tolist(),
        "breadth_max": max(breadth),
    }


def plotNetworkFromEdges(edges_list, **kwargs):
    G = nx.DiGraph()
    G.add_edges_from(edges_list)
    nx.draw_networkx(G, with_labels=False, node_size=50, **kwargs)
    return G


def convertNodeIds(row):
    nodes = row["nodes"]
    if nodes:
        node_dict = {id: (idx + 1) for idx, id in enumerate(nodes)}
        source = range(1, len(nodes) + 1)
        target = [node_dict.get(p, 0) for p in row["parents"]]
        return pd.Series(
            {
                "edges": list(map(list, zip(source, target))),
                "edges_dist": list(
                    np.divide(
                        np.subtract(np.abs(np.subtract(source, target)), 1),
                        np.subtract(source, 1),
                    )
                ),
                "no_time_problem": np.alltrue(np.greater(source, target)),
            }
        )


def get_cascade_edge_df(row):
    return pd.Series(
        {
            "root_id": row.root_id,
            "nodes": row.thread_node_id,
            "parents": row.thread_parent,
            "edges": list(zip(row.thread_node_id, row.thread_parent)),
        }
    )


def calculate_recent_root_distances(egde_list):
    cascade_vector, _ = np.histogram(
        egde_list, bins=30, range=(0, 1), normed=True, density=True
    )
    cascade_vector /= cascade_vector.sum()
    return pd.Series(
        {"recent_edge": cascade_vector[0], "root_edge": cascade_vector[-1]}
    )


def renameUserFeaturesColumns(user_stats):
    """
    This method renames the multi-level column dataframe containing user features. The
    dataframe is the result of an aggregation of groupby applying more than one function.
    The first level contains the aggregated field, while the second level has the function.
    Renaming by:
     1. adding a prefix to help sorting/grouping user features
     2. combining the metric field and the statistic applied
    
    @returns the user features dataframe with a single-level columns renamed.
    """
    return [
        "uf_{}_{}".format(
            user_stats.columns.levels[0][user_stats.columns.labels[0][i]],
            user_stats.columns.levels[1][user_stats.columns.labels[1][i]],
        )
        for i in range(len(user_stats.columns.labels[0]))
    ]


def calculateRedditUserFeatures(df):
    """
    Apply summary functions to the input dataframe in order to calculate user features.
    Each row contains info related to one submission:
        1. breadth_max      
        2. depth_max            
        3. first_level_comments 
        4. lifetime             
        5. root_user            
        6. size                 
        7. subreddit            
        8. recent_edge          
        9. root_edge  
        
    @return a dataframe with a user per row, where columns are the metric_statistic measured.
    """
    mean_and_median = ["mean", "median"]
    user_stats = df.groupby("root_user").agg(
        {
            "breadth_max": mean_and_median,
            "depth_max": mean_and_median,
            "first_level_comments": mean_and_median,
            "lifetime": mean_and_median,
            "size": mean_and_median + ["sum"],
            "recent_edge": mean_and_median,
            "root_edge": mean_and_median,
            "subreddit": ["nunique", "count"],
        }
    )

    user_stats.columns = renameUserFeaturesColumns(user_stats)

    user_stats.rename(
        {
            "uf_subreddit_count": "uf_num_posts",
            "uf_subreddit_nunique": "uf_num_subreddit",
        },
        axis=1,
        inplace=True,
    )

    return user_stats


def calculateTwitterUserFeatures(df):
    """
    Apply summary functions to the input dataframe in order to calculate user features.
    Each row contains info related to one submission:
        1. breadth_max      
        2. depth_max            
        3. first_level_comments 
        4. lifetime             
        5. root_user            
        6. size                 
        7. root_id            
        8. recent_edge          
        9. root_edge  
        
    @return a dataframe with a user per row, where columns are the metric_statistic measured.
    """
    mean_and_median = ["mean", "median"]
    user_stats = df.groupby("root_user").agg(
        {
            "breadth_max": mean_and_median,
            "depth_max": mean_and_median,
            "first_level_comments": mean_and_median,
            "lifetime": mean_and_median,
            "size": mean_and_median + ["sum"],
            "recent_edge": mean_and_median,
            "root_edge": mean_and_median,
            "root_id": ["count"],
        }
    )

    user_stats.columns = renameUserFeaturesColumns(user_stats)

    user_stats.rename({"uf_root_id_count": "uf_num_posts"}, axis=1, inplace=True)

    return user_stats


fields_to_change = ["edges", "thread_node_id", "thread_parent", "thread_user"]


def reverseInconsistentNodes(row):
    idxs = np.where(np.less_equal(*zip(*row["edges"])))[0]
    for idx in idxs:
        row["edges"][idx] = list(reversed(row["edges"][idx]))
        for col in fields_to_change[1:]:
            row[col][idx + 1], row[col][idx] = row[col][idx], row[col][idx + 1]
    return row


def checkActionType(row, data, id_dict):
    root = "?"
    parent = "?"
    provisory = None

    if row.is_quote:
        if row.is_quote_of_quote:
            result = "quote_of_quote"
            # as quote of an intermediary element, it has a partial parent. For sake of simplicity, treat as reply
            #             parent = row.is_quote
            provisory = row.is_quote
        elif row.is_reply:
            if row.is_retweet:
                result = "reply_of_retweet_of_quote"
            else:
                result = "reply_of_quote"
            # as a reply, get the direct parentID, but infer rootID
            parent = row.is_reply
        elif row.is_quote_of_reply:
            if row.is_retweet:
                result = "retweet_of_quote_of_reply"
                # as retweet of an intermediary element, it has a partial parent. For sake of simplicity, treat as reply
                #                 parent = row.is_retweet
                provisory = row.is_retweet
            else:
                result = "quote_of_reply"
                # as quote of an intermediary element, it has a partial parent. For sake of simplicity, treat as reply
                #                 parent = row.is_quote
                provisory = row.is_quote
        elif row.is_retweet:
            result = "retweet_of_quote"
            # as retweet of an intermediary element, it has a partial parent. For sake of simplicity, treat as reply
            #             parent = row.is_retweet
            provisory = row.is_retweet
        else:
            # defining root and parent for 'pure' quotes
            result = "quote"
            root = row.is_quote
    elif row.is_reply:
        if row.is_retweet:
            result = "reply_of_retweet"
        else:
            result = "reply"
        # as a reply, get the direct parentID, but infer rootID
        parent = row.is_reply
    elif row.is_retweet:
        if row.is_retweet_of_reply:
            result = "retweet_of_reply"
            # as retweet of an intermediary element, it has a partial parent. For sake of simplicity, treat as reply
            #             parent = row.is_retweet
            provisory = row.is_retweet
        elif row.is_retweet_of_quote:
            result = "retweet_of_quote"
            # as retweet of an intermediary element, it has a partial parent. For sake of simplicity, treat as reply
            #             parent = row.is_retweet
            provisory = row.is_retweet
        else:
            # retweet of an original simple tweet
            result = "retweet"
            root = row.is_retweet
    else:
        # simple original tweet
        result = "tweet"
        parent = row.nodeID
        root = row.nodeID

    # creating dict to add PNNL prefixes to nodeIDs
    id_dict[row.nodeID] = "t{}_{}".format(3 if result == "tweet" else 1, row.nodeID)
    return pd.Series(
        {
            "actionType": result,
            "parentID": parent,
            "rootID": root,
            "provisoryParent": provisory,
        }
    )


def generateCascadeSummary(row):
    row = row.sort_values("nodeTime")
    root_time = row.nodeTime.iloc[0]
    root_user = row.nodeUserID.iloc[0]
    if len(row) > 1:  # has any comment
        thread_node_id = row.nodeID.iloc[1:].tolist()
        thread_parent = row.parentID.iloc[1:].tolist()
        thread_user = row.nodeUserID.iloc[1:].tolist()
        thread_time = row.nodeTime.iloc[1:].tolist()
        thread_time_diff = np.subtract(thread_time, root_time)
        thread_action = row.actionType.iloc[1:].tolist()
        # check whether the cascade tree is broken
        if len(np.setdiff1d(thread_parent, thread_node_id)) != 1:
            print("Bronken Tree")
            print(row)
            return None
    else:
        thread_node_id = []
        thread_parent = []
        thread_time = []
        thread_time_diff = []
        thread_user = []
        thread_action = []

    return pd.Series(
        {
            "root_id": row.nodeID.iloc[0],
            "root_time": root_time,
            "root_user": root_user,
            "num_comments": len(thread_node_id),
            "num_users": len(set(thread_user)),
            "thread_node_id": thread_node_id,
            "thread_parent": thread_parent,
            "thread_user": thread_user,
            "thread_time": thread_time,
            "thread_time_diff": thread_time_diff,
            "title": row.text.iloc[0],
            "all_text": " &&!&& ".join(row.text),
            "thread_action": thread_action,
        }
    )
