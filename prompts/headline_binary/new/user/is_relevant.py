f"""Pattern: {hypothesis}
New headlines:
Headline 1: {headlines_0}
Headline 2: {headlines_1}
Answer:
"""

# add focus on differences between the pair
# but the old prompts are better on val data
#f"""Learned pattern: {hypothesis}
#Is the pattern relevant to the following pair of headlines?
#Headline 1: {headlines_0}
#Headline 2: {headlines_1}
#Think step by step.
#Step 1: Analyze the difference between "Headline 1" and "Headline 2".
#Step 2: Think about whether the pattern is relevant to the new pair of headlines.
#Step 3: Give your answer.
#"""