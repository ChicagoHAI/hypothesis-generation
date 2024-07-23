f"""Here are some previously generated patterns with some examples where it predicted which tweet will will be retweeted more.
{knn_info_prompt}
The first tweet: {first_text}
The second tweet: {second_text}
Which one of the two tweets will get more retweets?
Think step by step.
Step 1: Look at the new pair of tweets and compare them with the examples associated with each pattern.
Step 2: Find the set of examples that is closest to the given pair of tweets, and pick the pattern associated with that set of examples.
Step 3: Analyze the textual difference between the two tweets.
Step 4: Apply the picked pattern to the new pair of tweets. Based on that pattern, think about which one out of the pair of headlines will get more clicks.
Step 5: Give your final answer.
"""