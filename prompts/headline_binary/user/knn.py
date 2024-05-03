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
