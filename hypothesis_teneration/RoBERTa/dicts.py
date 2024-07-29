def reverse_dict(original_dict):
    reversed_dict = {}
    for key, value in original_dict.items():
        reversed_dict[value] = key
    return reversed_dict

SHOE_DICT = {
    'black': 0, 
    'blue': 1, 
    'green': 2, 
    'orange': 3, 
    'red': 4, 
    'white': 5
}

HOTEL_REVIEWS_DICT = {
    'truthful': 1,
    'deceptive': 0
}

HEADLINE_BINARY_DICT = {
    'headline 1':0,
    'headline 2':1
}
RETWEET_DICT = {
    'first':0,
    'second':1
}

LABEL_DICT = {
    'shoe': SHOE_DICT,
    'hotel_reviews': HOTEL_REVIEWS_DICT,
    'headline_binary':HEADLINE_BINARY_DICT,
    'retweet':RETWEET_DICT
}

PROMPT_NAME_DICT = {
    'shoe': 'appearance',
    'hotel_reviews': 'review',
    'headline_binary':'headline',
    'retweet':'tweets'
}