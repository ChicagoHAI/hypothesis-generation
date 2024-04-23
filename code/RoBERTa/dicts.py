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

SST_DICT = {
    'likely': 2,
    'unlikely': 1,
    'very likely': 3,
    'very unlikely': 0 
}

ORIGINAL_SST_DICT = {
    'very negative': 0,
    'negative': 1,
    'positive': 2,
    'very positive': 3
}

BINARY_SST_DICT = {
    'unlikely': 0,
    'likely': 1
}   

BINARY_ORIGINAL_SST_DICT = {
    'negative': 0,
    'positive': 1
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
    'sst': SST_DICT,
    'original_sst': ORIGINAL_SST_DICT,
    'binary_sst' : BINARY_SST_DICT,
    'binary_original_sst': BINARY_ORIGINAL_SST_DICT,
    'hotel_reviews': HOTEL_REVIEWS_DICT,
    'headline_binary':HEADLINE_BINARY_DICT,
    'retweet':RETWEET_DICT
}

PROMPT_NAME_DICT = {
    'shoe': 'appearance',
    'sst': 'sentence',
    'original_sst': 'sentence',
    'binary_sst' : 'sentence',
    'binary_original_sst': 'sentence',
    'hotel_reviews': 'review',
    'headline_binary':'headline',
    'retweet':'tweets'
}