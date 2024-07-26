f"""{f'Here are some examples of customers with certain features buying certain products:\n{observations}' if few_shot_flag else ''}
New customer: {info} is buying a pair of shoes, the shoes should be which color?
Answer:
"""