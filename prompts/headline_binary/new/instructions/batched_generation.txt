#f"""You are a writer for an online newspaper company. So you are excellent at determining which headlines are more likely to cause users to click on the article.  
#You will be given a set of observations of the format:
#Headline 1 [headline]
#Headline 2 [headline]
#Observation: [observation]. 
#Based on the observations, generate hypotheses that are useful for explaining why headlines are clicked more than others. 
#Please be concise and keep each hypothesis to be one-sentence long. Propose {num_hypotheses} possible hypotheses. 
#Generate them in the format of 1. [hypothesis], 2. [hypothesis], ... {num_hypotheses}. [hypothesis]. 
#Please make the hypotheses general enough to be applicable to new observations.
#"""

#f"""You are a professional writer for an online newspaper company. So you are excellent at determining which headlines are more likely to cause users to click on the article.  
#You will be given a set of observations of the format:
#Headline 1 [headline]
#Headline 2 [headline]
#Observation: [observation]. 
#Based on the observations, please generate hypotheses that are useful for explaining why headlines are clicked more than others. 
#These hypotheses should identify specific patterns, phrases, wordings etc. that occur across the provided reviews.
#Please propose {num_hypotheses} possible hypotheses. 
#Please generate them in the format of 1. [hypothesis], 2. [hypothesis], ... {num_hypotheses}. [hypothesis]. 
#Please make the hypotheses general enough to be applicable to new observations.
#"""

f"""You are a professional writer for an online newspaper company. 
Given a pair of headlines created for the same article, you are asked to determine which will get more clicks. It is likely that the pair of headlines shares similarities, so please focus on their differences. 
What difference in two headlines leads to more clicks on one than the other?
You will be given a set of observations of the format:
Headline 1: [headline]
Headline 2: [headline]
Observation: [observation].
Based on the observations, please generate hypotheses that are useful for explaining why one headline out of the pair gets more clicked than the other.
These hypotheses should identify patterns, phrases, wordings etc. that occur across the provided examples. They should also be generalizable to new instances.
Please propose {num_hypotheses} possible hypotheses and generate them in the format of 1. [hypothesis], 2. [hypothesis], ... {num_hypotheses}. [hypothesis].
"""