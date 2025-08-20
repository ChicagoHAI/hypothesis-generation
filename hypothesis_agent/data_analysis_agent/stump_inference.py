import re

from hypogenic.algorithm.inference import DefaultInference
from hypogenic.logger_config import LoggerConfig
from hypogenic.tasks import BaseTask
from hypogenic.algorithm.summary_information import (
    SummaryInformation,
)

from hypothesis_agent.data_analysis_agent.prompt import TestPrompt
import pandas as pd


class StumpInference(DefaultInference):
    def __init__(
            self,
            api,
            prompt_class: TestPrompt,
            train_data: pd.DataFrame,
            task: BaseTask,
            grouping_api,
    ):
        super().__init__(api, prompt_class, train_data, task)
        self.grouping_api = grouping_api

    @staticmethod
    def _parse_stump_response(response):
        """
        Parse the stump response to extract groups and their hypotheses.
        Returns a dictionary mapping group numbers to their hypotheses.
        """
        # Remove think tags if present
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.IGNORECASE | re.DOTALL)

        # Parse the response to extract groups and their hypotheses
        groups = {}
        current_group = None

        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Look for group indicators (e.g., "Group 1:" or "Group 1: <Description>")
            # Must start with "Group" followed by a number and colon
            group_match = re.match(r'^Group\s*(\d+)\s*:\s*(.*)', line, re.IGNORECASE)
            if group_match:
                current_group = int(group_match.group(1))
                current_condition = group_match.group(2).strip()
                
                if current_group not in groups:
                    groups[current_group] = {
                        'condition': current_condition,
                        'hypotheses': []
                    }
            elif current_group is not None and (line.startswith('-') or line.startswith('•')):
                # This is a hypothesis under the current group (indicated by '-' or '•')
                hypothesis = line.lstrip('- ').lstrip('• ').strip()
                if hypothesis:
                    groups[current_group]['hypotheses'].append(hypothesis)
            elif current_group is not None and re.match(r'^\d+\.', line):
                # This is a numbered hypothesis under the current group (e.g., "1. Hypothesis text")
                hypothesis = re.sub(r'^\d+\.\s*', '', line).strip()
                if hypothesis:
                    groups[current_group]['hypotheses'].append(hypothesis)

        # Log parsed results for debugging
        logger = LoggerConfig.get_logger("StumpParsing")
        logger.info("=== Parsed Stump Response ===")
        log_text = ''
        for group_num, group_info in sorted(groups.items()):
            log_text += f"Group {group_num}: {group_info['condition']}\n"
            for i, hypothesis in enumerate(group_info['hypotheses'], 1):
                log_text += f"  {i}. {hypothesis}\n"
        logger.info(log_text)
        logger.info("=============================")

        return groups

    def _create_stump(
            self,
            hyp_bank,
            cache_seed=None,
            **generate_kwargs,
    ):
        logger = LoggerConfig.get_logger("StumpCreation")
        prompt_input = self.prompt_class.create_stump(hyp_bank)
        response = self.grouping_api.generate(
            prompt_input,
            cache_seed=cache_seed,
            **generate_kwargs,
        )
        logger.info(f"Stump response: {response}")
        groups = self._parse_stump_response(response)
        return groups

    def _determine_groups_batched(
            self,
            data: pd.DataFrame,
            groups,
            cache_seed=None,
            max_concurrent=3,
            **generate_kwargs,
    ):
        """
        Determine which group each sample belongs to in batch.
        """
        logger = LoggerConfig.get_logger("SampleGroupDetermination")
        # Format group conditions for the prompt
        group_conditions = []
        for group_num, group_info in groups.items():
            condition = group_info['condition']
            group_conditions.append(f"Group {group_num}: {condition}")
        group_conditions_text = "\n".join(group_conditions)

        # Create prompts for all samples
        prompt_inputs = []
        for idx in range(len(data)):
            prompt_input = self.prompt_class.determine_group(group_conditions_text, data, idx)
            prompt_inputs.append(prompt_input)

        # Batch generate responses
        responses = self.api.batched_generate(
            prompt_inputs,
            cache_seed=cache_seed,
            max_concurrent=max_concurrent,
            **generate_kwargs,
        )
        # Extract group numbers from responses
        group_nums = []
        for response in responses:
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.IGNORECASE | re.DOTALL)
            response = response.strip()
            try:
                group_num = int(re.search(r'\d+', response).group())
                group_nums.append(group_num)
            except (ValueError, AttributeError):
                logger.error(f"Failed to parse group number from response: {response}, defaulting to group 1")

        return group_nums

    def _stump_batched_predict(
            self,
            data: pd.DataFrame,
            groups,
            cache_seed=None,
            max_concurrent=3,
            **generate_kwargs,
    ):
        """
        Make predictions using group-specific hypotheses with batched processing.
        """
        logger = LoggerConfig.get_logger("StumpBatchedPredict")
        # First, determine groups for all samples in batch
        group_nums = self._determine_groups_batched(
            data, groups, cache_seed, max_concurrent, **generate_kwargs
        )

        # Create prompts for all samples with their respective group condition and hypotheses
        prompt_inputs = []
        sample_indices = []

        for idx in range(len(data)):
            group_num = group_nums[idx]

            if group_num in groups:
                group_hypotheses = groups[group_num]['hypotheses']
                group_condition = groups[group_num]['condition']
                # Convert to the format expected by multiple_hypotheses_inference
                # Create a dictionary mapping hypothesis text to SummaryInformation
                hyp_dict = {}
                for i, hypothesis_text in enumerate(group_hypotheses):
                    hyp_dict[hypothesis_text] = SummaryInformation()

                # Create prompt for this sample with group-specific hypotheses
                prompt_input = self.prompt_class.stump_predict(
                    hyp_dict, data, idx, group_condition
                )
                prompt_inputs.append(prompt_input)
                sample_indices.append(idx)
            else:
                logger.error(f"Group {group_num} not found in groups.")

        # Batch generate predictions
        responses = self.api.batched_generate(
            prompt_inputs,
            cache_seed=cache_seed,
            max_concurrent=max_concurrent,
            **generate_kwargs,
        )

        # Extract predictions and actual labels
        predictions = []
        actual_labels = []

        for i, response in enumerate(responses):
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.IGNORECASE | re.DOTALL)
            prediction = self.task.extract_label(response)
            predictions.append(prediction)
            actual_labels.append(data[self.task.label_name][sample_indices[i]])

        # Log group statistics and prediction distribution
        self._log_group_statistics(groups, group_nums, predictions, sample_indices)

        return predictions, actual_labels

    @staticmethod
    def _log_group_statistics(groups, group_nums, predictions, sample_indices):
        """
        Log statistics for each group including sample count and prediction distribution.
        """
        logger = LoggerConfig.get_logger("GroupStatistics")
        log_text = ''

        # Create a mapping from sample index to group number
        sample_to_group = {}
        for i, group_num in enumerate(group_nums):
            sample_to_group[sample_indices[i]] = group_num
        
        # Analyze each group
        for group_num, group_info in groups.items():
            group_condition = group_info['condition']
            
            # Find samples belonging to this group
            group_samples = [idx for idx, g_num in sample_to_group.items() if g_num == group_num]
            sample_count = len(group_samples)
            
            # Get predictions for samples in this group
            group_predictions = []
            for sample_idx in group_samples:
                if sample_idx in sample_indices:
                    pred_idx = sample_indices.index(sample_idx)
                    if pred_idx < len(predictions):
                        group_predictions.append(predictions[pred_idx])
            
            # Count prediction distribution
            pred_distribution = {}
            for pred in group_predictions:
                pred_distribution[pred] = pred_distribution.get(pred, 0) + 1
            
            # Format prediction distribution string
            pred_dist_str = ", ".join([f"{label}: {count}" for label, count in pred_distribution.items()])
            
            # Log group statistics
            log_text += f"Group {group_num}: {group_condition}\n"
            log_text += f"Sample count: {sample_count}\n"
            log_text += f"Prediction distribution: {pred_dist_str}\n"
            log_text += "-" * 50 +"\n"

        logger.info(log_text)

    def run_inference_final(
            self,
            data,
            hyp_bank,
            cache_seed=None,
            max_concurrent=3,
            **generate_kwargs,
    ):
        # Create stump from hypotheses
        groups = self._create_stump(
            hyp_bank,
            cache_seed=cache_seed,
            **generate_kwargs,
        )

        # Run inference using group-specific hypotheses
        return self._stump_batched_predict(
            data,
            groups,
            cache_seed=cache_seed,
            max_concurrent=max_concurrent,
            **generate_kwargs,
        )