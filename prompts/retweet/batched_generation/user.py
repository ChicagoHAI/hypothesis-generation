f"""We made some observations:
{observations}
Generate hypotheses that are useful for predicting which tweet out of a pair of tweets is more likely to be retweeted.
Please note that the paired tweets are about the same content and are posted by the same user, so you should focus on the wording difference between the two tweets in each pair.
Please propose {num_hypotheses} possible hypotheses. 
Please generate them in the format of: 
1. [hypothesis] 
2. [hypothesis] 
... 
{num_hypotheses}. [hypothesis].
Proposed hypotheses:
"""