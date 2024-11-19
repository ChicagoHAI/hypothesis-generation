from typing import List, Dict, Any, Tuple
from hypogenic.prompt import BasePrompt
from hypogenic.LLM_wrapper.base import LLMWrapper
from hypogenic.algorithm.summary_information import SummaryInformation
from hypogenic.logger_config import LoggerConfig
import os
import re


def extract_relevance_results(response: str) -> bool:
    logger = LoggerConfig.get_logger("HypoAgent - Utils")
    response = response.lower()
    result = re.findall(r"final answer: ?(yes|no)", response)

    if len(result) == 0:
        logger.warning(f"No final answer found in response:\n{response}")
        return False

    if result[-1] == "yes":
        return True
    else:
        return False


def check_hypothesis_relevance(
    prompt_class: BasePrompt,
    api,
    hypothesis: str,
    test_data,
    test_idx,
    cache_seed=None,
    max_concurrent=32,
    **generate_kwargs,
):
    prompt_input = prompt_class.is_relevant(
        hypotheses_dict={"hypothesis": hypothesis},
        test_data=test_data,
        test_idx=test_idx,
    )

    response = api.generate(
        prompt_input,
        cache_seed=cache_seed,
        max_concurrent=max_concurrent,
        **generate_kwargs,
    )

    if "yes" in response.lower():
        return True
    else:
        return False


def batched_check_hypothesis_relevance(
    prompt_class: BasePrompt,
    api,
    hypotheses: Dict[str, Any],
    test_data,
    test_idx: List[int],
    cache_seed=None,
    max_concurrent=32,
    **generate_kwargs,
) -> Dict[str, Dict[int, bool]]:

    relevance_results = {}
    prompt_inputs = []
    hyp_list = list(hypotheses.keys())
    for i in range(0, len(hyp_list)):
        hypothesis = hyp_list[i]
        relevance_results[hypothesis] = {}
        for j in range(0, len(test_idx)):
            prompt_inputs.append(
                prompt_class.is_relevant({hypothesis: []}, test_data, test_idx[j])
            )
    responses = api.batched_generate(
        prompt_inputs,
        max_concurrent,
        cache_seed,
        **generate_kwargs,
    )
    responses = responses[::-1]

    for i in range(0, len(hyp_list)):
        hypothesis = hyp_list[i]
        for j in range(0, len(test_idx)):
            response = responses.pop(-1)
            response = response.lower()
            tmp_response = response
            if "final answer:" in response:
                response = response[
                    response.index("final answer:") + len("final answer:") :
                ]
                response = response[:5]
                response = response.lower()
            else:
                print("############response format error##############")
                print(response)

            if "yes" in response and "no" in response:
                if "yes or no" in response:
                    relevance = False
                else:
                    print(
                        f'The response should not contain both "yes" and "no". Response: {response}'
                    )
                    relevance = False
            elif "yes" in response:
                relevance = True
            else:
                relevance = False

            relevance_results[hypothesis][test_idx[j]] = relevance
    return relevance_results


def check_hypothesis_pair_repetition(
    prompt_class: BasePrompt,
    api,
    hyp_bank: Dict[str, Any],
    cache_seed=None,
    max_concurrent=32,
    **generate_kwargs,
):
    hyp_list = list(hyp_bank.keys())
    prompt_input = prompt_class.check_hypothesis_pair_repetition(hyp_list)
    response = api.generate(
        prompt_input,
        cache_seed=cache_seed,
        max_concurrent=max_concurrent,
        **generate_kwargs,
    )

    if "yes" in response.lower():
        return True
    else:
        return False


def batched_check_hypotheses_repetition(
    prompt_class: BasePrompt,
    api,
    hyp_bank: Dict[str, Any],
    cache_seed=None,
    max_concurrent=32,
    **generate_kwargs,
) -> Dict[str, Dict[str, bool]]:
    hyp_list = list(hyp_bank.keys())
    prompt_inputs = []
    for i in range(0, len(hyp_list)):
        for j in range(i + 1, len(hyp_list)):
            prompt_inputs.append(
                prompt_class.check_hypothesis_pair_repetition(
                    [hyp_list[i], hyp_list[j]]
                )
            )
    responses = api.batched_generate(
        prompt_inputs,
        max_concurrent,
        cache_seed,
        **generate_kwargs,
    )
    responses = responses[::-1]
    repetition_mat = {}

    for i in range(0, len(hyp_list)):
        repetition_mat[hyp_list[i]] = {}
    for i in range(0, len(hyp_list)):
        hyp = hyp_list[i]
        for j in range(i + 1, len(hyp_list)):
            response = responses.pop(-1)
            response = response.lower()
            tmp_response = response
            if "final answer:" in response:
                response = response[
                    response.index("final answer:") + len("final answer:") :
                ]
                response = response[:5]
                response = response.lower()
            else:
                print("############response format error##############")
                print(response)

            if "yes" in response and "no" in response:
                if "yes or no" in response:
                    repetition = False
                else:
                    print(
                        f'The response should not contain both "yes" and "no". Response: {response}'
                    )
                    repetition = False
            elif "yes" in response:
                repetition = True
            else:
                repetition = False
            repetition_mat[hyp][hyp_list[j]] = repetition
            repetition_mat[hyp_list[j]][hyp] = repetition

    return repetition_mat


def multiple_hypotheses_remove_repetition(
    prompt_class: BasePrompt,
    api,
    hyp_bank: Dict[str, SummaryInformation],
    cache_seed=None,
    max_concurrent=32,
    **generate_kwargs,
) -> Dict[str, SummaryInformation]:
    """
    Remove repeating hypotheses
    Among repeating ones we keep the one with highest training acc
    """

    repetition_mat = batched_check_hypotheses_repetition(
        prompt_class,
        api,
        hyp_bank,
        cache_seed,
        max_concurrent,
        **generate_kwargs,
    )
    # re-rank by acc
    sorted_hyp_list = sorted(hyp_bank, key=lambda x: hyp_bank[x].acc, reverse=True)
    unique_hyp_list = []
    for i in range(0, len(sorted_hyp_list)):
        have_repetition = False
        for hyp in unique_hyp_list:
            if repetition_mat[sorted_hyp_list[i]][hyp] == True:
                have_repetition = True
        if have_repetition == False:
            unique_hyp_list.append(sorted_hyp_list[i])
            print("added: ", sorted_hyp_list[i])
            print()
        else:
            print("skipped: ", sorted_hyp_list[i])
            print()

    unique_hyp_bank = {}
    for hyp in unique_hyp_list:
        unique_hyp_bank[hyp] = hyp_bank[hyp]
    return unique_hyp_bank