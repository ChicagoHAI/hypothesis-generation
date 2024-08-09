import re
from ..register import Register

extract_label_register = Register("extract_label")


@extract_label_register.register("headline_binary")
def headline_binary_extract_label(text):
    if text is None:
        return "other"
    text = text.lower()
    pattern = r"answer:\s+(headline 1|headline 2|other)"
    match = re.search(pattern, text.lower())
    """
    if "Answer:" in text:
        text = text[text.index("Answer:") + len("Answer:"):]
    for x in self.label_classes:
        if x.lower() in text.lower():
        return x
    """
    if match:
        answer = match.group(1)
        if answer == "headline 1":
            return "headline 1"
        elif answer == "headline 2":
            return "headline 2"
        else:
            return "other"
    return "other"


@extract_label_register.register("hotel_reviews")
def hotel_reviews_extract_label(text):
    if text is None:
        return "other"

    # only keep the part after "Final answer:"
    text = text.lower()
    """
    if "final answer:" in text:
        text = text[text.index("final answer:") + len("final answer:"):]



    if "label:" in text:
        # only keep the part after "label:"
        text = text[text.index("label:") + len("label:"):]
    """

    pattern = r"final answer:\s+(truthful|deceptive|other)"

    match = re.search(pattern, text.lower())
    if match:
        answer = match.group(1)
        if answer == "truthful":
            return "truthful"
        elif answer == "deceptive":
            return "deceptive"
        else:
            return "other"

    return "other"


@extract_label_register.register("retweet")
def retweet_extract_label(text):
    """
    `text` follows the format "the <label> tweet got more retweets"
    """
    if text is None:
        return "other"
    text = text.lower()
    pattern = r"answer: the (\w+) tweet"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return "other"


@extract_label_register.register("shoe")
def extract_label(text):
    if text is None:
        return "other"

    pattern = r"final answer:\s+(white|red|orange|green|blue|black)"

    match = re.search(pattern, text.lower())
    if match:
        answer = match.group(1)
        if answer in ["white", "red", "orange", "green", "blue", "black"]:
            return answer
        else:
            return "other"

    return "other"
