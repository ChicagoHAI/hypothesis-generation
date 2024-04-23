f"""Here are some previously generated patterns with some examples where it predicted correctly for whether a hotel review is deceptive or truthful.
{knn_info_prompt}
{review}
Look at the new hotel review and compare it with the set of examples associated with each provided pattern. 
Find the set of examples that is the most similar to the new hotel review, pick and repeat the pattern associated with that set of examples.
Remember to follow the format:
Please give your final answer in the following format:
Reasoning for choosing pattern: reason,
Chosen pattern: Pattern <number>.

Answer:
"""