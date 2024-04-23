f"""Here are some previously generated patterns with some examples where it predicted which tweet will will be retweeted more.
{knn_info_prompt}
The first tweet: {first_text}
The second tweet: {second_text}
Which one of the two tweets will get more retweets?
Think step by step.
Step 1: Analyze the difference between the first tweet and the second tweet.
Step 2: Find the set of examples that is closest to the given pair of tweets, and pick the pattern associated with that set of examples.
"""