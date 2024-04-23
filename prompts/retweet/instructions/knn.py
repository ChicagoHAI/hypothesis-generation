f"""You are a social media expert.
Given a pair of tweets, you are asked to predict which tweet will be retweeted more.
Please note that the paired tweets are about the same content and are posted by the same user, so you should focus on the wording difference between the two tweets.
From past experiences, you learned some patterns.
You should apply a learned pattern to a pair of tweets and determine which one will get more retweets. 
For each pattern, you will also see a couple of examples that worked for each pattern.
Please choose a pattern. To do this, look at the examples associated with each pattern, and find which set of the examples are closest to the given pair of tweets. 
Please choose the pattern corresponding to that set of examples.
The answer for the higher retweets should be of the form "the _ tweet" where _ is either first or second. 
Please give your final answer in the following format:
Reasoning for choosing pattern: reason,
Chosen pattern: pattern,
Reasoning for choice of prediction: reason,
Final Answer: answer
"""