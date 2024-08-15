from ...logger_config import LoggerConfig

logger = LoggerConfig.get_logger("HypoGenic - Generation")


def extract_hypotheses(text, num_hypotheses):
    """
    Given a response with hypotheses, we want to take all of them out from the text.
    This function might need to be adjusted depending on the prompt and the
    robust-ness of the LM

    Parameters:
        text: the text with the hypotheses enclosed
        num_hypotheses: the number of hypotheses that you want
    """
    import re

    # The regex to extract the hypotheses - matches to a numbered list
    pattern = re.compile(r"\d+\.\s(.+?)(?=\d+\.\s|\Z)", re.DOTALL)
    logger.info(f"Text provided {text}")
    hypotheses = pattern.findall(text)

    # Here, either there are no hypotheses, or you didn't match them correctly
    if len(hypotheses) == 0:
        logger.info("No hypotheses are generated.")
        return []

    # this is a bit sketchy
    if len(hypotheses) != num_hypotheses:
        logger.info(f"Expected {num_hypotheses} hypotheses, but got {len(hypotheses)}.")

    for i in range(len(hypotheses)):
        hypotheses[i] = hypotheses[i].strip()

    return hypotheses[:num_hypotheses]
