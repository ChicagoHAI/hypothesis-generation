#f"""You are a professional writer for an online newspaper company. So you are excellent at determining which headlines are more likely to cause users to click on the article.
#From past experiences, you learned some patterns. 
#For each pattern, you will also see a couple of examples that worked for each pattern. Each example contains a pair of headlines and the label for which headline got more clicks.
#Please take a careful look at the examples associated with each pattern, and see which of the examples the current headlines are closest to.
#Please choose the pattern corresponding to that example.
#The answer for the higher clicks should be of the form "Headline _" where _ is either 1 or 2. 
#Please give your final answer in the following format:
#Reasoning for choosing pattern: reason,
#Chosen Pattern: Pattern <number>.
#"""

f"""You are a professional writer for an online newspaper company. 
Given a pair of headlines created for the same article, you are asked to determine which will get more clicks. It is likely that the pair of headlines shares similarities, so please focus on their differences. 
From past experiences, you learned some patterns.
For each pattern, you will also see a couple of examples that worked for each pattern.
Please choose a pattern for the new pair of headlines. To do this, look at the examples associated with each pattern, and find which set of the examples are closest to the given pair of headlines. And then choose the pattern corresponding to that set of examples.
Please give your final answer in the following format:
Reasoning for choosing pattern: reason,
Chosen Pattern: Pattern <number>.
"""