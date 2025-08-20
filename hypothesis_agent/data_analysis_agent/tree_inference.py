import re

from hypogenic.algorithm.inference import DefaultInference
from hypogenic.logger_config import LoggerConfig
from hypogenic.tasks import BaseTask
from hypogenic.algorithm.summary_information import (
    SummaryInformation,
)

from hypothesis_agent.data_analysis_agent.prompt import TestPrompt
import pandas as pd


class TreeNode:
    def __init__(self, groups=None, hypotheses=None, children=None, is_leaf=False, path=None):
        self.groups = groups or {}  # {group_num: {condition, hypotheses, examples}}
        self.hypotheses = hypotheses or []
        self.children = children or {} # {index}
        self.is_leaf = is_leaf
        self.path = path or [] # [condition1, condition2, ...]


class TreeInference(DefaultInference):
    def __init__(
            self,
            api,
            prompt_class: TestPrompt,
            train_data: pd.DataFrame,
            task: BaseTask,
    ):
        super().__init__(api, prompt_class, train_data, task)


    # --------------------------------
    # Tree create
    # --------------------------------
    @staticmethod
    def _parse_split_decision(response):
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.IGNORECASE | re.DOTALL)
        response = response.strip().upper()
        if any(keyword in response for keyword in ["NO_SPLIT"]):
            return "NO_SPLIT"
        elif any(keyword in response for keyword in ["SPLIT"]):
            return "SPLIT"
        else:
            raise ValueError(f"Could not parse split decision from response: {response}")

    @staticmethod
    def _parse_tree_response(response):
        response = re.sub(r'<think>.*?</think>', '', response, flags=re.IGNORECASE | re.DOTALL)
        groups = {}
        current_group = None
        current_section = None  # 'condition', 'hypotheses', or 'examples'

        lines = response.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue

            group_match = re.match(r'^Group\s*(\d+)[:.]?\s*(.*)', line, re.IGNORECASE)
            if group_match:
                current_group = int(group_match.group(1))
                current_condition = group_match.group(2).strip()
                if current_group not in groups:
                    groups[current_group] = {
                        'condition': current_condition,
                        'hypotheses': [],
                        'examples': []
                    }
                current_section = 'condition'
                continue

            if re.match(r'^Examples?[:.]?\s*$', line, re.IGNORECASE):
                if current_group is not None:
                    current_section = 'examples'
                continue

            if current_group is not None and re.match(r'^\d+\.', line):
                current_section = 'hypotheses'
                hypothesis = re.sub(r'^\d+\.\s*', '', line).strip()
                if hypothesis:
                    groups[current_group]['hypotheses'].append(hypothesis)
                continue

            if current_group is not None and current_section == 'examples' and re.match(r'^\d+\.', line):
                example = re.sub(r'^\d+\.\s*', '', line).strip()
                if example:
                    groups[current_group]['examples'].append(example)
                continue

            if current_group is not None and current_section == 'hypotheses' and (line.startswith('-') or line.startswith('•')):
                hypothesis = line.lstrip('- ').lstrip('• ').strip()
                if hypothesis:
                    groups[current_group]['hypotheses'].append(hypothesis)
            elif current_group is not None and current_section == 'examples' and (line.startswith('-') or line.startswith('•')):
                example = line.lstrip('- ').lstrip('• ').strip()
                if example:
                    groups[current_group]['examples'].append(example)

        return groups

    def _tree_split(
            self,
            hyp_bank,
            cache_seed=None,
            max_depth=5,
            current_depth=0,
            current_path=None,
            **generate_kwargs,
    ):
        logger = LoggerConfig.get_logger("TreeSplit")
        if current_depth >= max_depth:
            logger.info(f"Reached max depth {max_depth}, creating leaf node")
            return TreeNode(is_leaf=True, hypotheses=hyp_bank, path=current_path or [])

        logger.info(f"=== Tree Split Decision at Depth {current_depth} ===")
        logger.info(f"Current path: {current_path or []}")
        prompt_input = self.prompt_class.tree_split_decision(hypotheses_dict=hyp_bank, current_depth=current_depth, current_path=current_path)
        response = self.api.generate(
            prompt_input,
            cache_seed=cache_seed,
            **generate_kwargs,
        )
        logger.info(f"--- LLM Output for Split Decision ---")
        logger.info(f"Response: {response}")
        split_decision = self._parse_split_decision(response)

        if split_decision == "NO_SPLIT":
            logger.info(f"LLM decided not to split at depth {current_depth}, creating leaf node")
            return TreeNode(is_leaf=True, hypotheses=hyp_bank, path=current_path or [])
        elif split_decision == "SPLIT":
            logger.info(f"LLM decided to split at depth {current_depth}, generating groups")
            split_prompt_input = self.prompt_class.tree_split(hypotheses_dict=hyp_bank, current_path=current_path)
            split_response = self.api.generate(
                split_prompt_input,
                cache_seed=cache_seed,
                **generate_kwargs,
            )
            logger.info(f"--- LLM Output for Tree Split ---")
            logger.info(f"Response: {split_response}")
            groups = self._parse_tree_response(split_response)

            node = TreeNode()
            node.groups = groups
            node.hypotheses = hyp_bank
            node.path = current_path or []

            for group_num, group_info in groups.items():
                child_path = (current_path or []) + [group_info['condition']]
                logger.info(f"--- Recursing to Group {group_num} ---")
                logger.info(f"Child condition path: {child_path}")
                
                # Create a new hypothesis bank that includes both hypotheses and examples
                child_hyp_bank = group_info['hypotheses'].copy()
                if group_info.get('examples'):
                    # Add examples as additional context for child nodes
                    child_hyp_bank.extend(group_info['examples'])
                
                child_node = self._tree_split(
                    child_hyp_bank,  # Pass both hypotheses and examples
                    cache_seed=cache_seed,
                    max_depth=max_depth,
                    current_depth=current_depth + 1,
                    current_path=child_path,
                    **generate_kwargs
                )
                node.children[group_num] = child_node
            return node
        else:
            logger.warning(f"Could not parse split decision at depth {current_depth}, defaulting to leaf node")
            return TreeNode(is_leaf=True, hypotheses=hyp_bank, path=current_path or [])

    # --------------------------------
    # Tree update (validation and refinement)
    # --------------------------------
    def _visualize_tree_log(self, node, depth=0):
        if node.is_leaf:
            # Leaf node
            result = f"LEAF NODE:\n"
            if node.hypotheses:
                result += "Hypotheses:\n"
                for i, hyp in enumerate(node.hypotheses, 1):
                    result += f"  {i}. {hyp}\n"
            return result
        else:
            # Internal node
            result = f"INTERNAL NODE:\n"
            if node.hypotheses:
                result += "General Hypotheses:\n"
                for i, hyp in enumerate(node.hypotheses, 1):
                    result += f"  {i}. {hyp}\n"
            
            result += "Groups:\n"
            for group_num, group_info in node.groups.items():
                result += f"  Group {group_num}: {group_info['condition']}\n"
                
                if group_info.get('hypotheses'):
                    result += "    Refined Hypotheses:\n"
                    for i, hyp in enumerate(group_info['hypotheses'], 1):
                        result += f"      {i}. {hyp}\n"
                
                if group_info.get('examples'):
                    result += "    Examples:\n"
                    for i, example in enumerate(group_info['examples'], 1):
                        result += f"      {i}. {example}\n"
                
                # Recurse to children
                if group_num in node.children:
                    child_result = self._visualize_tree(node.children[group_num], depth + 1)
                    # Indent child result
                    child_lines = child_result.split('\n')
                    indented_child = '\n'.join(f"    {line}" if line else "" for line in child_lines)
                    result += f"    Subtree:\n{indented_child}\n"
            
            return result

    def _visualize_tree(self, node, depth=0):
        if node.is_leaf:
            # Leaf node - show hypotheses
            result = f"LEAF NODE:\n"
            if node.hypotheses:
                result += "Hypotheses:\n"
                for i, hyp in enumerate(node.hypotheses, 1):
                    result += f"  {i}. {hyp}\n"
            return result
        else:
            # Internal node
            result = f"INTERNAL NODE:\n"
            
            result += "Groups:\n"
            for group_num, group_info in node.groups.items():
                result += f"  Group {group_num}: {group_info['condition']}\n"
                if group_info.get('examples'):
                    result += "    Examples:\n"
                    for i, example in enumerate(group_info['examples'], 1):
                        result += f"      {i}. {example}\n"
                
                # Recurse to children
                if group_num in node.children:
                    child_result = self._visualize_tree(node.children[group_num], depth + 1)
                    # Indent child result
                    child_lines = child_result.split('\n')
                    indented_child = '\n'.join(f"    {line}" if line else "" for line in child_lines)
                    result += f"    Subtree:\n{indented_child}\n"
            
            return result

    def _tree_validation(self, tree_root, original_hypotheses, cache_seed=None, **generate_kwargs):
        logger = LoggerConfig.get_logger("TreeValidation")
        logger.info("=== Tree Validation Phase ===")
        tree_structure = self._visualize_tree(tree_root)
        prompt_input = self.prompt_class.tree_validation(original_hypotheses=original_hypotheses,
                                                         tree_structure=tree_structure)
        validation_response = self.api.generate(
            prompt_input,
            cache_seed=cache_seed,
            **generate_kwargs,
        )
        logger.info("--- LLM Output for Tree Validation ---")
        logger.info(f"Validation response: {validation_response}")
        validation_passed = self._check_validation_result(validation_response)
        if not validation_passed:
            logger.warning(f"Validation failed.")
        return validation_response, validation_passed

    @staticmethod
    def _check_validation_result(validation_response):
        response_lower = validation_response.lower()
        pass_keywords = ['VALID']
        fail_keywords = ['INVALID']
        if any(keyword in response_lower for keyword in fail_keywords):
            return False
        if any(keyword in response_lower for keyword in pass_keywords):
            return True
        return False

    def _tree_refinement(self, tree_root, original_hypotheses, validation_analysis, cache_seed=None, **generate_kwargs):
        logger = LoggerConfig.get_logger("TreeRefinement")
        logger.info("=== Tree Refinement Phase ===")
        tree_structure = self._visualize_tree(tree_root)
        prompt_input = self.prompt_class.tree_refinement(original_hypotheses=original_hypotheses,
                                                         tree_structure=tree_structure,
                                                         validation_analysis=validation_analysis)
        refinement_response = self.api.generate(
            prompt_input,
            cache_seed=cache_seed,
            **generate_kwargs,
        )
        logger.info("--- LLM Output for Tree Refinement ---")
        logger.info(f"Refinement response: {refinement_response}")
        logger.info("--- Parsing Improved Tree Structure ---")
        improved_tree_root = self._parse_and_build_improved_tree(refinement_response, original_hypotheses)
        return improved_tree_root

    def _parse_and_build_improved_tree(self, refinement_response, original_hypotheses):
        logger = LoggerConfig.get_logger("TreeRefinement")
        try:
            logger.info("--- Parsing Complete Tree Structure ---")
            tree_structure = self._parse_complete_tree_structure(refinement_response)
            if not tree_structure:
                logger.warning("Could not parse complete tree structure from refinement response")
                return None
            logger.info("--- Building Complete Improved Tree Structure ---")
            improved_root = self._build_tree_from_structure(tree_structure, original_hypotheses, [])
            if improved_root:
                logger.info("=== Successfully Built Complete Improved Tree Structure ===")
                return improved_root
            else:
                logger.error("Failed to build tree from parsed structure")
                return None
        except Exception as e:
            logger.error(f"Error building improved tree: {e}")
            return None

    def _parse_complete_tree_structure(self, refinement_response):
        """Parse the structured tree format from LLM response"""
        response = re.sub(r'<think>.*?</think>', '', refinement_response, flags=re.IGNORECASE | re.DOTALL)
        lines = response.strip().split('\n')
        
        # Parse the structured tree format
        tree_structure = self._parse_structured_tree(lines, 0, len(lines))
        return tree_structure

    def _parse_structured_tree(self, lines, start_idx, end_idx):
        """Parse the new structured tree format"""
        groups = {}
        i = start_idx
        
        while i < end_idx:
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            
            # Look for group start
            group_match = re.match(r'^Group\s*(\d+):\s*(.*)', line, re.IGNORECASE)
            if group_match:
                group_num = int(group_match.group(1))
                condition = group_match.group(2).strip()
                
                # Initialize new group
                groups[group_num] = {
                    'condition': condition,
                    'hypotheses': [],
                    'examples': [],
                    'subgroups': {},
                    'is_leaf': True  # Default to leaf
                }
                
                # Parse group content
                j = i + 1
                while j < end_idx:
                    next_line = lines[j].strip()
                    if not next_line:
                        j += 1
                        continue
                    
                    # Check if we've reached the next group or section
                    if re.match(r'^Group\s*\d+:', next_line, re.IGNORECASE):
                        break
                    if next_line.startswith('INTERNAL NODE:') or next_line.startswith('LEAF NODE:'):
                        break
                    
                    # Parse refined hypotheses
                    if next_line.startswith('Refined Hypotheses:'):
                        j += 1
                        while j < end_idx:
                            hyp_line = lines[j].strip()
                            if not hyp_line:
                                j += 1
                                continue
                            
                            # Check for end of hypotheses section
                            if (hyp_line.startswith('Examples:') or 
                                hyp_line.startswith('Subtree:') or
                                re.match(r'^Group\s*\d+:', hyp_line, re.IGNORECASE) or
                                hyp_line.startswith('INTERNAL NODE:') or
                                hyp_line.startswith('LEAF NODE:')):
                                break
                            
                            # Extract hypothesis
                            hyp_match = re.match(r'^\s*\d+\.\s*(.+)', hyp_line)
                            if hyp_match:
                                hypothesis = hyp_match.group(1).strip()
                                groups[group_num]['hypotheses'].append(hypothesis)
                            
                            j += 1
                        continue
                    
                    # Parse examples
                    if next_line.startswith('Examples:'):
                        j += 1
                        while j < end_idx:
                            ex_line = lines[j].strip()
                            if not ex_line:
                                j += 1
                                continue
                            
                            # Check for end of examples section
                            if (ex_line.startswith('Subtree:') or
                                re.match(r'^Group\s*\d+:', ex_line, re.IGNORECASE) or
                                ex_line.startswith('INTERNAL NODE:') or
                                ex_line.startswith('LEAF NODE:')):
                                break
                            
                            # Extract example
                            ex_match = re.match(r'^\s*\d+\.\s*(.+)', ex_line)
                            if ex_match:
                                example = ex_match.group(1).strip()
                                groups[group_num]['examples'].append(example)
                            
                            j += 1
                        continue
                    
                    # Parse subtree
                    if next_line.startswith('Subtree:'):
                        j += 1
                        subtree_start = j
                        
                        # Find subtree end
                        subtree_end = self._find_subtree_end(lines, subtree_start, end_idx)
                        
                        # Parse nested subgroups
                        nested_groups = self._parse_structured_tree(lines, subtree_start, subtree_end)
                        groups[group_num]['subgroups'] = nested_groups
                        groups[group_num]['is_leaf'] = False
                        
                        j = subtree_end
                        continue
                    
                    j += 1
                
                i = j - 1  # Adjust for the loop increment
            
            i += 1
        
        return groups

    @staticmethod
    def _find_subtree_end(lines, start_idx, end_idx):
        """Find the end of a subtree section"""
        i = start_idx
        
        while i < end_idx:
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            
            # Check for end of subtree
            if (line.startswith('Group') or
                line.startswith('INTERNAL NODE:') or
                line.startswith('LEAF NODE:') or
                (line.startswith('Group') and ':' in line)):
                return i

            i += 1

        return end_idx

    def _build_tree_from_structure(self, tree_structure, original_hypotheses, current_path):
        logger = LoggerConfig.get_logger("TreeRefinement")

        if not tree_structure:
            return None

        root = TreeNode()
        root.groups = tree_structure
        root.hypotheses = original_hypotheses
        root.path = current_path
        root.children = {}

        for group_num, group_info in tree_structure.items():
            if group_info.get('is_leaf', False):
                # Create leaf node
                leaf_node = TreeNode(
                    is_leaf=True,
                    hypotheses=group_info['hypotheses'],
                    path=current_path + [group_info['condition']]
                )
                root.children[group_num] = leaf_node
                logger.info(f"Created leaf node for Group {group_num}")
            else:
                # Create internal node with subgroups
                if group_info.get('subgroups'):
                    internal_node = TreeNode()
                    internal_node.groups = group_info['subgroups']
                    internal_node.hypotheses = group_info['hypotheses']
                    internal_node.path = current_path + [group_info['condition']]
                    internal_node.children = {}

                    # Recursively build subtree for subgroups
                    for subgroup_num, subgroup_info in group_info['subgroups'].items():
                        if subgroup_info.get('is_leaf', False):
                            subgroup_node = TreeNode(
                                is_leaf=True,
                                hypotheses=subgroup_info['hypotheses'],
                                path=internal_node.path + [subgroup_info['condition']]
                            )
                            internal_node.children[subgroup_num] = subgroup_node
                        else:
                            # Handle deeper nesting recursively
                            if subgroup_info.get('subgroups'):
                                # This is an internal node with its own subgroups
                                nested_node = self._build_tree_from_structure(
                                    subgroup_info['subgroups'], 
                                    subgroup_info.get('hypotheses', []),
                                    internal_node.path + [subgroup_info['condition']]
                                )
                                if nested_node:
                                    nested_node.groups = subgroup_info['subgroups']
                                    nested_node.hypotheses = subgroup_info.get('hypotheses', [])
                                    nested_node.path = internal_node.path + [subgroup_info['condition']]
                                    internal_node.children[subgroup_num] = nested_node
                                    logger.info(f"Created nested internal node for Group {group_num} -> Subgroup {subgroup_num}")
                                else:
                                    # Fallback: treat as leaf node
                                    logger.warning(f"Failed to build nested node for subgroup {subgroup_num}, treating as leaf")
                                    subgroup_node = TreeNode(
                                        is_leaf=True,
                                        hypotheses=subgroup_info.get('hypotheses', []),
                                        path=internal_node.path + [subgroup_info['condition']]
                                    )
                                    internal_node.children[subgroup_num] = subgroup_node
                            else:
                                # Fallback: treat as leaf node
                                logger.warning(f"Subgroup {subgroup_num} has no subgroups but is not marked as leaf, treating as leaf")
                                subgroup_node = TreeNode(
                                    is_leaf=True,
                                    hypotheses=subgroup_info.get('hypotheses', []),
                                    path=internal_node.path + [subgroup_info['condition']]
                                )
                                internal_node.children[subgroup_num] = subgroup_node

                    root.children[group_num] = internal_node
                    logger.info(
                        f"Created internal node for Group {group_num} with {len(group_info['subgroups'])} subgroups")
                else:
                    # Fallback: treat as leaf node
                    leaf_node = TreeNode(
                        is_leaf=True,
                        hypotheses=group_info['hypotheses'],
                        path=current_path + [group_info['condition']]
                    )
                    root.children[group_num] = leaf_node
                    logger.info(f"Created fallback leaf node for Group {group_num}")

        return root

    # --------------------------------
    # Tree inference
    # --------------------------------
    def _tree_batched_predict(
            self,
            data: pd.DataFrame,
            tree_root: TreeNode,
            cache_seed=None,
            max_concurrent=3,
            **generate_kwargs,
    ):
        logger = LoggerConfig.get_logger("TreeBatchedPredict")
        sample_paths = self._determine_paths_batched(
            data, tree_root, cache_seed, max_concurrent, **generate_kwargs
        )

        path_groups = {}
        for idx, path in enumerate(sample_paths):
            path_key = tuple(path)
            if path_key not in path_groups:
                path_groups[path_key] = []
            path_groups[path_key].append(idx)

        all_predictions = [None] * len(data)
        all_actual_labels = [None] * len(data)

        logger.info("=== Leaf Node Analysis ===")
        for path, indices in path_groups.items():
            if path:
                leaf_node = self._get_leaf_node_by_path(tree_root, path)
                if leaf_node and leaf_node.is_leaf:
                    leaf_data = data.iloc[indices].reset_index(drop=True)
                    
                    batch_predictions = self._leaf_batch_predict(
                        leaf_data, leaf_node, cache_seed, max_concurrent, **generate_kwargs
                    )
                    logger.info(f"Predictions: {batch_predictions}")
                    
                    for i, pred in enumerate(batch_predictions):
                        all_predictions[indices[i]] = pred
                        all_actual_labels[indices[i]] = data[self.task.label_name][indices[i]]
                else:
                    raise RuntimeError(f"Resolved path {path} did not lead to a valid leaf node.")
            else:
                raise RuntimeError("Empty path encountered for some samples.")
        
        return all_predictions, all_actual_labels

    @staticmethod
    def _extract_group_num(response_text: str, allowed_groups: set[int]):
        cleaned = re.sub(r'<think>.*?</think>', '', str(response_text), flags=re.IGNORECASE | re.DOTALL).strip()
        candidates = re.findall(r'\d+', cleaned)
        for c in candidates:
            try:
                num = int(c)
                if num in allowed_groups:
                    return num
            except ValueError:
                continue
        return None

    def _determine_paths_batched(self, data, tree_root, cache_seed=None, max_concurrent=3, **generate_kwargs):
        logger = LoggerConfig.get_logger("DeterminePathsBatched")
        current_nodes = [tree_root] * len(data)
        current_paths = [[] for _ in range(len(data))]

        layer_count = 0
        while any(not node.is_leaf for node in current_nodes):
            layer_count += 1
            logger.info(f"=== Processing Layer {layer_count} ===")
            non_leaf_indices = [i for i, node in enumerate(current_nodes) if not node.is_leaf]
            if not non_leaf_indices:
                break
            node_groups = {}
            for idx in non_leaf_indices:
                node = current_nodes[idx]
                node_key = id(node)
                if node_key not in node_groups:
                    node_groups[node_key] = {'node': node, 'indices': []}
                node_groups[node_key]['indices'].append(idx)
            for node_key, group_info in node_groups.items():
                node = group_info['node']
                indices = group_info['indices']
                group_conditions = []
                for group_num, group_detail in node.groups.items():
                    condition = group_detail['condition']
                    group_conditions.append(f"Group {group_num}: {condition}")
                allowed_list = sorted(list(node.groups.keys()))
                allowed_set = set(allowed_list)
                allowed_text = ", ".join(str(x) for x in allowed_list)
                group_conditions_text = "\n".join(group_conditions) + f"\nValid group numbers: {allowed_text}. Answer must be ONE of these numbers ONLY."
                logger.info(f"Group conditions: {group_conditions}")

                prompt_inputs = []
                for idx in indices:
                    sample_data = data.iloc[[idx]]
                    sample_data = sample_data.reset_index(drop=True)
                    prompt_input = self.prompt_class.internal_inference(group_conditions_text, sample_data, 0)
                    prompt_inputs.append(prompt_input)

                responses = self.api.batched_generate(
                    prompt_inputs,
                    cache_seed=cache_seed,
                    max_concurrent=max_concurrent,
                    **generate_kwargs,
                )
                logger.info(f"--- LLM Outputs for Group Assignment ---")
                for i, response in enumerate(responses):
                    sample_idx = indices[i]
                    logger.info(f"Sample {sample_idx} response: {response}")

                parsed_groups: dict[int, int | None] = {}
                invalid_local_positions = []  # position within indices
                for i, response in enumerate(responses):
                    sample_idx = indices[i]
                    gnum = self._extract_group_num(response, allowed_set)
                    if gnum is None:
                        invalid_local_positions.append(i)
                        parsed_groups[sample_idx] = None
                        logger.warning(f"Sample {sample_idx}: Failed to parse group from response: '{response}'")
                    else:
                        parsed_groups[sample_idx] = gnum
                if invalid_local_positions:
                    logger.info(f"Retrying {len(invalid_local_positions)} samples with stronger constraints")
                    retry_inputs = []
                    for pos in invalid_local_positions:
                        idx_global = indices[pos]
                        sample_data = data.iloc[[idx_global]].reset_index(drop=True)
                        retry_text = group_conditions_text + f"\nChoose strictly one from [{allowed_text}] and output ONLY the numeral."
                        retry_inputs.append(self.prompt_class.internal_inference(retry_text, sample_data, 0))
                    
                    retry_responses = self.api.batched_generate(
                        retry_inputs,
                        cache_seed=cache_seed,
                        max_concurrent=max_concurrent,
                        **generate_kwargs,
                    )

                    logger.info(f"--- LLM Retry Outputs ---")
                    for j, resp in enumerate(retry_responses):
                        pos = invalid_local_positions[j]
                        sample_idx = indices[pos]
                        logger.info(f"Sample {sample_idx} retry response: {resp}")
                    
                    for j, resp in enumerate(retry_responses):
                        pos = invalid_local_positions[j]
                        sample_idx = indices[pos]
                        gnum = self._extract_group_num(resp, allowed_set)
                        parsed_groups[sample_idx] = gnum
                        if gnum is not None:
                            logger.info(f"Sample {sample_idx}: Retry successful, assigned to group {gnum}")
                        else:
                            logger.warning(f"Sample {sample_idx}: Retry failed, response: '{resp}'")

                group_distribution = {}
                for sample_idx, gnum in parsed_groups.items():
                    if gnum is None:
                        raise RuntimeError(
                            f"Failed to get a valid group among {allowed_list} for sample index {sample_idx}."
                        )

                    if gnum in node.children:
                        current_nodes[sample_idx] = node.children[gnum]
                        current_paths[sample_idx].append(gnum)
                        group_distribution[gnum] = group_distribution.get(gnum, 0) + 1
                        logger.info(f"Sample {sample_idx}: Path updated to {current_paths[sample_idx]}")
                    else:
                        raise RuntimeError(
                            f"Predicted group {gnum} not in node.children. Path so far: {current_paths[sample_idx]}"
                        )

                logger.info(f"Layer {layer_count} group distribution: {group_distribution}")
                logger.info(f"Updated paths: {[current_paths[i] for i in indices]}")

        logger.info(f"=== Path Determination Complete ===")
        logger.info(f"Total layers processed: {layer_count}")
        
        return current_paths

    @staticmethod
    def _get_leaf_node_by_path(tree_root, path):
        current_node = tree_root
        for group_num in path:
            if not current_node.is_leaf and group_num in current_node.children:
                current_node = current_node.children[group_num]
            else:
                return None
        return current_node

    def _leaf_batch_predict(self, leaf_data, leaf_node, cache_seed=None, max_concurrent=3, **generate_kwargs):
        logger = LoggerConfig.get_logger("LeafBatchPredict")

        logger.info(f"=== Leaf Node Prediction ===")
        hyp_dict = {}
        for hypothesis_text in leaf_node.hypotheses:
            hyp_dict[hypothesis_text] = SummaryInformation()
        prompt_inputs = []
        for idx in range(len(leaf_data)):
            sample_data = leaf_data.iloc[[idx]]
            sample_data = sample_data.reset_index(drop=True)
            prompt_input = self.prompt_class.multiple_hypotheses_inference_with_path(
                hyp_dict=hyp_dict, sample_data=sample_data, condition_path=leaf_node.path
            )
            prompt_inputs.append(prompt_input)
        responses = self.api.batched_generate(
            prompt_inputs,
            cache_seed=cache_seed,
            max_concurrent=max_concurrent,
            **generate_kwargs,
        )
        logger.info(f"--- LLM Outputs for Leaf Prediction ---")
        for i, response in enumerate(responses):
            logger.info(f"Sample {i} response: {response}")
        predictions = []
        for i, response in enumerate(responses):
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.IGNORECASE | re.DOTALL)
            prediction = self.task.extract_label(response)
            predictions.append(prediction)
            logger.info(f"Sample {i}: Extracted prediction: {prediction}")
        return predictions


    # --------------------------------
    # Interface for running tree inference
    # --------------------------------
    def run_inference_final(
            self,
            data,
            hyp_bank,
            cache_seed=None,
            max_concurrent=3,
            max_depth=5,
            max_iterations=5,
            **generate_kwargs,
    ):
        logger = LoggerConfig.get_logger("RunInferenceFinal")
        logger.info("=== Starting Hierarchical Tree Inference ===")
        
        logger.info("--- Phase 1: Building Decision Tree ---")
        tree_root = self._tree_split(
            hyp_bank=hyp_bank,
            cache_seed=cache_seed,
            max_depth=max_depth,
            **generate_kwargs,
        )

        # Display the built tree structure after Phase 1
        logger.info("=== Phase 1 Complete: Built Decision Tree ===")
        logger.info("Tree Structure:")
        tree_structure = self._visualize_tree(tree_root)
        logger.info(tree_structure)
        logger.info("=" * 80)

        logger.info("--- Phase 2: Tree Validation and Refinement ---")
        current_tree_root = tree_root
        iteration_count = 0
        validation_passed = False

        while iteration_count < max_iterations and not validation_passed:
            iteration_count += 1
            logger.info(f"--- Iteration {iteration_count}/{max_iterations} ---")

            # Show current tree structure before validation
            logger.info(f"Current tree structure before validation):")
            current_tree_structure = self._visualize_tree(current_tree_root)
            logger.info(current_tree_structure)

            validation_analysis, validation_passed = self._tree_validation(
                tree_root=current_tree_root, original_hypotheses=hyp_bank, cache_seed=cache_seed, **generate_kwargs
            )
            if validation_passed:
                logger.info(f"=== Tree Validation Passed at Iteration {iteration_count} ===")
                break
            else:
                logger.info(f"=== Tree Validation Failed at Iteration {iteration_count} ===")
                if iteration_count >= max_iterations:
                    logger.warning(f"Reached maximum iterations ({max_iterations}), using current tree")
                    break
                logger.info(f"--- Proceeding to Tree Refinement for Iteration {iteration_count} ---")
                refined_tree_root = self._tree_refinement(
                    tree_root=current_tree_root, original_hypotheses=hyp_bank, validation_analysis=validation_analysis, cache_seed=cache_seed, **generate_kwargs
                )
                current_tree_root = refined_tree_root

                # Show refined tree structure after refinement
                logger.info(f"Refined tree structure (after iteration {iteration_count}):")
                refined_tree_structure = self._visualize_tree(current_tree_root)
                logger.info(refined_tree_structure)
                logger.info("-" * 60)
        logger.info("--- Tree Validation and Refinement Complete ---")
        logger.info(f"Total iterations: {iteration_count}")
        if validation_passed:
            logger.info("Tree validation passed successfully")
        else:
            logger.warning("Tree validation did not pass, but proceeding with current tree")

        logger.info("--- Phase 3: Running Tree Inference ---")
        predictions, actual_labels = self._tree_batched_predict(
            data,
            current_tree_root,
            cache_seed=cache_seed,
            max_concurrent=max_concurrent,
            **generate_kwargs,
        )

        logger.info("=== Hierarchical Tree Inference Complete ===")
        logger.info(f"Prediction distribution: {pd.Series(predictions).value_counts().to_dict()}")
        return predictions, actual_labels
