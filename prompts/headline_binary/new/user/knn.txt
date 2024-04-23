f"""Here are some previously generated patterns with some examples where it predicted which one of the pair of headlines got more clicks.
{knn_info_prompt}
Which one out of the following pair of headlines will get more clicks?
Headline 1: {headlines_0}
Headline 2: {headlines_1}
Think step by step.
Step 1: Look at the new pair of headlines and compare them with the examples associated with each pattern.
Step 2: Find the set of examples that is closest to the given pair of headlines, and pick the pattern associated with that set of examples.
Step 3: Apply the picked pattern to the new pair of headlines. Based on that pattern, think about which one out of the pair of headlines will get more clicks.
Step 4: Give your final answer.
"""

# add the focus on difference
#f"""Here are some previously generated patterns with some examples where they predicted the proper headline with more clicks.
#{knn_info_prompt}
#Which one out of the following new pair of headlines will get more clicks?
#Headline 1: {headlines_0}
#Headline 2: {headlines_1}
#Think step by step.
#Step 1: Analyze the difference between "Headline 1" and "Headline 2".
#Step 2: Find the set of examples that is closest to the given pair of headlines, and pick the pattern associated with that set of examples.
#Step 3: Apply the picked pattern to the new pair of headlines. Based on that pattern, think about which one out of the pair of headlines will get more clicks.
#Step 4: Give your final answer.
#"""