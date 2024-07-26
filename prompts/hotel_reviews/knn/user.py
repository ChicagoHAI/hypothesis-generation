f"""Here are some previously generated patterns with some example where it predicted correctly if a hotel review is deceptive or truthful.
{knn_info_prompt}
New hotel review:
{info}
Is this review deceptive or truthful?
Think step-by-step.
Step 1: Look at the new hotel review and compare it with the set of examples associated with each provided pattern. 
Step 2: Find the set of examples that is the most similar to the new hotel review, pick and repeat the pattern associated with that set of examples.
Step 3: Apply the pattern you picked to the new hotel review and predict whether the new hotel review is deceptive or truthful.
Step 4: Give your final answer.
Answer:
"""