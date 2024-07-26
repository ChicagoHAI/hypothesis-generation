f"""Here are some previously generated patterns with some examples where they predicted the proper headline with more clicks.
{knn_info_prompt}
New pair of headlines:
{info}
Think step by step.
Step 1: Analyze the difference between "Headline 1" and "Headline 2".
Step 2: Find the set of examples that is closest to the given pair of headlines, and pick the pattern associated with that set of examples.
"""