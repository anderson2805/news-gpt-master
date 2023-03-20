# load universal sentence encoder large at tensorflow_hub
from typing import List, Dict
import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow_hub as hub
import warnings
warnings.filterwarnings("ignore")


embed = hub.load('./model/universal-sentence-encoder-large/5')


def similar_contents(contentlists: List[str]) -> Dict[int, List[int]]:
    """
    Given a list of content descriptions, computes pairwise similarity between them and groups them based on a threshold.

    Args:
    - contentlists (list of str): A list of content descriptions.

    Returns:
    - similar_dict (dict): A dictionary where the keys are the indices of the content descriptions and the values are
      lists of indices of other content descriptions that are similar to the content description at the key index.

    Example:
    >>> contentlists = ['This is a sample text.', 'This text is similar to the previous one.', 'This is another text.']
    >>> similar_contents(contentlists)
    {0: [0, 1], 1: [0, 1], 2: [2]}

    """
    embeddings = embed(contentlists)  # type: ignore
    similar_dict = {}
    for index, row in enumerate(np.inner(embeddings, embeddings)):
        similar = []
        for sim_index, item in enumerate(row):
            if item >= 0.75:
                similar.append(sim_index)
        similar_dict[index] = similar
    return similar_dict


# group articles with similar contentdescription, and return a dataframe.
def group_articles(contents: pd.DataFrame) -> pd.DataFrame:
    """
    Groups articles with similar content descriptions and returns a dataframe.

    Args:
    - contents (list of dict): A list of dictionaries where each dictionary represents an article and has the following keys:
        - 'contentdescription' (str): The content description of the article.
        - 'publisheddate' (str): The date the article was published.
        - 'duplicates' (int): The number of duplicates of the article.

    Returns:
    - filteredDf (pandas DataFrame): A dataframe where each row represents an article and has the following columns:
        - 'contentdescription' (str): The content description of the article or the concatenated content descriptions of a group of similar articles.
        - 'publisheddate' (str): The date the article was published.
        - 'duplicates' (int): The number of duplicates of the article or the sum of duplicates of a group of similar articles.
        - 'group' (int): The group ID of the article if it belongs to a group or -1 if it does not belong to a group.
        - 'startdate' (str): The earliest published date of the articles in the same group as the article.
        - 'latestdate' (str): The latest published date of the articles in the same group as the article.

    Example:
    >>> contents = [{'contentdescription': 'This is a sample text.', 'publisheddate': '2022-01-01', 'duplicates': 1},
                    {'contentdescription': 'This text is similar to the previous one.', 'publisheddate': '2022-01-02', 'duplicates': 1},
                    {'contentdescription': 'This is another text.', 'publisheddate': '2022-01-03', 'duplicates': 1}]
    >>> group_articles(contents)
      contentdescription  publisheddate  duplicates  group   startdate   latestdate
    0  This is a sample text.\nThis text is similar...  2022-01-01          2      0  2022-01-01  2022-01-02
    2                                This is another text.  2022-01-03          1     -1  2022-01-03  2022-01-03

    """
    filteredDf = pd.DataFrame(contents)
    similar_dict = similar_contents(filteredDf['contentdescription'].tolist())
    # Create a new column to store the group ID of each article
    filteredDf['group'] = -1
    filteredDf['startdate'] = filteredDf['publisheddate']
    filteredDf['latestdate'] = filteredDf['publisheddate']
    # Initialize a group ID counter
    group_id = 0
    skipped = []
    # Iterate through each row of the dataframe
    for index, row in filteredDf.iterrows():
        # If the article has not been assigned to a group yet
        all_similar_index = []
        if row['group'] == -1 and index not in skipped:

            # Create a new group and add the current article to it
            filteredDf.loc[index, 'group'] = group_id
            group_articles = [index]

            # Find all similar articles and add them to the group
            for similar_index in similar_dict[index]:
                all_similar_index += similar_dict[similar_index]

            for similar_index in list(set(all_similar_index)):
                if similar_index not in skipped and filteredDf.loc[similar_index, 'group'] == -1:
                    filteredDf.loc[similar_index, 'group'] = group_id
                    group_articles.append(similar_index)

            # Concatenate the content descriptions of all articles in the group
            group_description = '\n'.join(
                filteredDf.loc[group_articles, 'contentdescription'].tolist())
            group_duplicates = filteredDf.loc[group_articles, 'duplicates'].sum()
            
            # Update the content description of the first article in the group
            filteredDf.loc[index, 'contentdescription'] = group_description
            filteredDf.loc[index, 'duplicates'] = group_duplicates

            # Set earlier and latest reported dates
            filteredDf.loc[index, 'startdate'] = filteredDf.loc[group_articles,
                                                                'publisheddate'].min()
            filteredDf.loc[index, 'latestdate'] = filteredDf.loc[group_articles,
                                                                'publisheddate'].max()

            # Remove all other articles in the group from the dataframe
            filteredDf.drop(group_articles[1:], inplace=True)
            skipped += (group_articles)

            # Increment the group ID counter
            group_id += 1
    return filteredDf.sort_values(by=['duplicates'], ascending=False)