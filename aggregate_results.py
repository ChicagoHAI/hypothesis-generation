#!/usr/bin/env python3
"""
Script to aggregate results from hypothesis generation experiments.

Usage:
    python aggregate_results.py --model MODEL_FOLDER --tasks TASK1,TASK2,... --methods METHOD1,METHOD2,...

Example:
    python aggregate_results.py --model "Qwen/Qwen2.5-72B-Instruct" --tasks "shoe_two_level/simple,admission/level_1" --methods "zero_shot_baseline,hypogenic"
"""

import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys

def get_method_results(methods_data: Dict[str, Any], method_name: str) -> tuple[Optional[float], Optional[float]]:
    """Extract accuracy and f1 for a given method from the methods data."""
    if method_name not in methods_data:
        return None, None

    method_data = methods_data[method_name]

    # Handle different result structures
    if isinstance(method_data, dict):
        # For methods like io_prompting and io_refinement that have a "best" field
        if "best" in method_data:
            best_data = method_data["best"]
            accuracy = best_data.get("test_accuracy") or best_data.get("accuracy")
            f1 = best_data.get("test_f1") or best_data.get("f1")
        else:
            # For simple methods with direct accuracy/f1
            accuracy = method_data.get("accuracy")
            f1 = method_data.get("f1")

        return accuracy, f1

    return None, None

def load_result_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """Load and parse a JSON result file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load {file_path}: {e}")
        return None

def aggregate_results(model_folder: str, task_folders: List[str], method_names: List[str],
                     results_dir: str = "results", seeds: List[int] = [42]) -> List[Dict[str, Any]]:
    """
    Aggregate results for given model, tasks, and methods.

    Args:
        model_folder: Model folder name (e.g., "Qwen/Qwen2.5-72B-Instruct")
        task_folders: List of task folder names (e.g., ["shoe_two_level/simple", "admission/level_1"])
        method_names: List of method names to extract
        results_dir: Root results directory
        seeds: List of seeds to check

    Returns:
        List of dictionaries with aggregated results
    """
    results = []
    results_path = Path(results_dir)

    for task_folder in task_folders:
        for seed in seeds:
            # Check for IND and OOD results
            for data_type in ["IND", "OOD"]:
                combined_file = results_path / task_folder / model_folder / f"combined_results_{data_type}_seed_{seed}.json"

                data = load_result_file(combined_file)
                if data is None:
                    # Record missing results
                    for method_name in method_names:
                        results.append({
                            "Method": method_name,
                            "Dataset": f"{task_folder}_{data_type}_seed_{seed}",
                            "Task": task_folder,
                            "Data_Type": data_type,
                            "Seed": seed,
                            "Accuracy": "N/A",
                            "F1": "N/A",
                            "Model": model_folder
                        })
                    continue

                methods_data = data.get("methods", {})

                for method_name in method_names:
                    accuracy, f1 = get_method_results(methods_data, method_name)

                    results.append({
                        "Method": method_name,
                        "Dataset": f"{task_folder}_{data_type}_seed_{seed}",
                        "Task": task_folder,
                        "Data_Type": data_type,
                        "Seed": seed,
                        "Accuracy": accuracy if accuracy is not None else "N/A",
                        "F1": f1 if f1 is not None else "N/A",
                        "Model": model_folder
                    })

    return results

def save_results_to_csv(results: List[Dict[str, Any]], output_file: str):
    """Save results to CSV file."""
    if not results:
        print("No results to save.")
        return

    # Define column order
    fieldnames = ["Method", "Dataset", "Task", "Data_Type", "Seed", "Accuracy", "F1", "Model"]

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Aggregate experimental results")
    parser.add_argument("--model", required=True, help="Model folder name")
    parser.add_argument("--tasks", required=True, help="Comma-separated list of task folders")
    parser.add_argument("--methods", required=True, help="Comma-separated list of method names")
    parser.add_argument("--seeds", default="42", help="Comma-separated list of seeds (default: 42)")
    parser.add_argument("--results-dir", default="results", help="Results directory (default: results)")
    parser.add_argument("--output", help="Output CSV file (default: results_MODEL.csv)")

    args = parser.parse_args()

    # Parse arguments
    task_folders = [task.strip() for task in args.tasks.split(",")]
    method_names = [method.strip() for method in args.methods.split(",")]
    seeds = [int(seed.strip()) for seed in args.seeds.split(",")]

    # Generate output filename if not provided
    if args.output:
        output_file = args.output
    else:
        model_safe = args.model.replace("/", "_").replace(" ", "_")
        output_file = f"results_{model_safe}.csv"

    print(f"Aggregating results for:")
    print(f"  Model: {args.model}")
    print(f"  Tasks: {task_folders}")
    print(f"  Methods: {method_names}")
    print(f"  Seeds: {seeds}")
    print(f"  Output: {output_file}")
    print()

    # Aggregate results
    results = aggregate_results(
        model_folder=args.model,
        task_folders=task_folders,
        method_names=method_names,
        results_dir=args.results_dir,
        seeds=seeds
    )

    # Save to CSV
    save_results_to_csv(results, output_file)

    # Print summary
    total_entries = len(results)
    available_entries = len([r for r in results if r["Accuracy"] != "N/A"])
    print(f"\nSummary:")
    print(f"  Total entries: {total_entries}")
    print(f"  Available entries: {available_entries}")
    print(f"  Missing entries: {total_entries - available_entries}")

if __name__ == "__main__":
    main()