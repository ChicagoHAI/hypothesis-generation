#!/bin/bash

# Configuration
MODEL_NAME="qwen/qwen3-32b"
# MODEL_NAME="gpt-4.1-mini"
# MODEL_NAME="meta-llama/llama-3.1-70b-instruct"

TASKS=(
    "deceptive_reviews"
    "with_heuristics/deceptive_reviews"
    "llamagc_detect"
    "with_heuristics/llamagc_detect"
    "gptgc_detect"
    "with_heuristics/gptgc_detect"
    "persuasive_pairs"
    "with_heuristics/persuasive_pairs"
    "dreaddit"
    "with_heuristics/dreaddit"
    "headline_binary"
    "with_heuristics/headline_binary"
    "retweet"
    "with_heuristics/retweet"
    "journal_same/same_journal_health"
    "with_heuristics/journal_same/same_journal_health"
    "journal_same/same_journal_nips"
    "with_heuristics/journal_same/same_journal_nips"
    "journal_same/same_journal_radiology"
    "with_heuristics/journal_same/same_journal_radiology"
)
METHODS=(
    "few_shot_gen"
    "literature_only"
    "hypogenic"
    "hyporefine"

)
SEED=42

# Convert arrays to comma-separated strings
TASKS_STR=$(IFS=,; echo "${TASKS[*]}")
METHODS_STR=$(IFS=,; echo "${METHODS[*]}")

# Generate output filename
MODEL_SAFE=$(echo "$MODEL_NAME" | sed 's/\//_/g')
OUTPUT_FILE="results_${MODEL_SAFE}_seed_${SEED}.csv"

echo "Running aggregation with:"
echo "  Model: $MODEL_NAME"
echo "  Tasks: $TASKS_STR"
echo "  Methods: $METHODS_STR"
echo "  Seed: $SEED"
echo "  Output: $OUTPUT_FILE"
echo

# Run the aggregation
python aggregate_results.py \
    --model "$MODEL_NAME" \
    --tasks "$TASKS_STR" \
    --methods "$METHODS_STR" \
    --seeds "$SEED" \
    --output "$OUTPUT_FILE"

echo "Done! Results saved to: $OUTPUT_FILE"