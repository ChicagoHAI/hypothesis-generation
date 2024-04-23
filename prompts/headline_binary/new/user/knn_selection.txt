# Rosa: I feel like choosing a pattern based on example similarities wouldn't work well for headline.
#f"""Here are some previously generated patterns with some example where it predicted the proper headline with more clicks.
#{knn_info_prompt}
#New headlines:
#Headline 1: {headlines_0}
#Headline 2: {headlines_1}
#Look at the new headlines and compare it with the set of examples associated with each provided pattern. 
#Find the set of examples that is the most similar to the headlines, pick and repeat the pattern associated with that set of examples.
#Remember to follow the format:
#Please give your final answer in the following format:
#Reasoning for choosing pattern: reason,
#Chosen Pattern: Pattern <number>.
#Answer:
#"""

f"""Here are some previously generated patterns with some examples where they predicted the proper headline with more clicks.
{knn_info_prompt}
New pair of headlines:
Headline 1: {headlines_0}
Headline 2: {headlines_1}
Think step by step.
Step 1: Analyze the difference between "Headline 1" and "Headline 2".
Step 2: Find the set of examples that is closest to the given pair of headlines, and pick the pattern associated with that set of examples.
"""