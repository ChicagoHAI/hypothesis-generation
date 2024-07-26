f"""{f'We have seen some hotel reviews:\n{observations}' if few_shot_flag else ''}
New hotel review:
{info}
Is this hotel review truthful or deceptive?
Answer:
"""