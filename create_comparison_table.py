#!/usr/bin/env python3
import pandas as pd
import numpy as np
from pathlib import Path

def load_all_results():
    """Load all CSV result files and combine them."""
    csv_files = list(Path('.').glob('results_*.csv'))
    all_data = []

    for file in csv_files:
        df = pd.read_csv(file)
        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)

def categorize_methods(df):
    """Add columns to categorize methods by heuristics usage."""
    df = df.copy()

    # Determine if method uses heuristics based on dataset name
    df['uses_heuristics'] = df['Dataset'].str.contains('with_heuristics/')

    return df

def aggregate_results(df):
    """Aggregate results by method, model, data_type, and heuristics usage."""

    # Group by the key dimensions and calculate mean accuracy and F1
    grouped = df.groupby(['Model', 'Method', 'Data_Type', 'uses_heuristics']).agg({
        'Accuracy': 'mean',
        'F1': 'mean'
    }).reset_index()

    return grouped

def create_comparison_table(aggregated_df):
    """Create the final comparison table."""

    # Get unique models and methods
    models = sorted(aggregated_df['Model'].unique())
    methods = sorted(aggregated_df['Method'].unique())
    data_types = sorted(aggregated_df['Data_Type'].unique())

    results = {}

    for model in models:
        model_data = aggregated_df[aggregated_df['Model'] == model]
        results[model] = {}

        for data_type in data_types:
            dt_data = model_data[model_data['Data_Type'] == data_type]
            results[model][data_type] = {}

            for method in methods:
                method_data = dt_data[dt_data['Method'] == method]

                # Get results without heuristics
                without_heur = method_data[method_data['uses_heuristics'] == False]
                # Get results with heuristics
                with_heur = method_data[method_data['uses_heuristics'] == True]

                results[model][data_type][method] = {
                    'without': {
                        'accuracy': without_heur['Accuracy'].iloc[0] if len(without_heur) > 0 else None,
                        'f1': without_heur['F1'].iloc[0] if len(without_heur) > 0 else None
                    },
                    'with': {
                        'accuracy': with_heur['Accuracy'].iloc[0] if len(with_heur) > 0 else None,
                        'f1': with_heur['F1'].iloc[0] if len(with_heur) > 0 else None
                    }
                }

    return results

def print_comparison_table(results):
    """Print the comparison table in the requested format."""

    models = list(results.keys())
    data_types = list(next(iter(results.values())).keys())

    for model in models:
        print(f"\n{model}:")
        print("=" * 50)

        # Print headers dynamically based on available data types
        header_line = f"{'Method':<15}"
        subheader_line = f"{'':<15}"

        for dt in data_types:
            header_line += f" {dt:<17}"
            subheader_line += f" {'Acc':<8} {'F1':<8}"

        print(header_line)
        print(subheader_line)
        print("-" * (15 + 18 * len(data_types)))

        methods = list(results[model][data_types[0]].keys())

        for method in methods:
            # Print method without heuristics
            line = f"{method:<15}"
            for dt in data_types:
                without = results[model][dt][method]['without']
                acc = f"{without['accuracy']:.3f}" if without['accuracy'] is not None else "N/A"
                f1 = f"{without['f1']:.3f}" if without['f1'] is not None else "N/A"
                line += f" {acc:<8} {f1:<8}"
            print(line)

            # Print method with heuristics
            line = f"*+heuristics   "
            for dt in data_types:
                with_heur = results[model][dt][method]['with']
                acc_h = f"{with_heur['accuracy']:.3f}" if with_heur['accuracy'] is not None else "N/A"
                f1_h = f"{with_heur['f1']:.3f}" if with_heur['f1'] is not None else "N/A"
                line += f" {acc_h:<8} {f1_h:<8}"
            print(line)
            print()

def save_results_to_csv(aggregated_df):
    """Save the aggregated results to a CSV file for further analysis."""

    # Pivot the data to have separate columns for with/without heuristics
    pivot_df = aggregated_df.pivot_table(
        index=['Model', 'Method', 'Data_Type'],
        columns='uses_heuristics',
        values=['Accuracy', 'F1'],
        aggfunc='first'
    ).reset_index()

    # Flatten column names
    pivot_df.columns = [f"{col[0]}_{col[1]}" if col[1] != '' else col[0]
                       for col in pivot_df.columns]

    # Rename columns for clarity
    column_mapping = {
        'Accuracy_False': 'Accuracy_without_heuristics',
        'Accuracy_True': 'Accuracy_with_heuristics',
        'F1_False': 'F1_without_heuristics',
        'F1_True': 'F1_with_heuristics'
    }

    pivot_df = pivot_df.rename(columns=column_mapping)

    # Calculate improvements where both values exist
    if 'Accuracy_with_heuristics' in pivot_df.columns and 'Accuracy_without_heuristics' in pivot_df.columns:
        pivot_df['Accuracy_improvement'] = (pivot_df['Accuracy_with_heuristics'] -
                                           pivot_df['Accuracy_without_heuristics'])

    if 'F1_with_heuristics' in pivot_df.columns and 'F1_without_heuristics' in pivot_df.columns:
        pivot_df['F1_improvement'] = (pivot_df['F1_with_heuristics'] -
                                     pivot_df['F1_without_heuristics'])

    pivot_df.to_csv('comparison_results.csv', index=False)
    print(f"\nDetailed results saved to comparison_results.csv")

def main():
    print("Loading all result files...")
    df = load_all_results()

    print(f"Found {len(df)} total results across {df['Model'].nunique()} models")
    print(f"Data types: {sorted(df['Data_Type'].unique())}")
    print(f"Methods: {sorted(df['Method'].unique())}")

    print("Categorizing methods...")
    df = categorize_methods(df)

    print("Aggregating results...")
    aggregated_df = aggregate_results(df)

    print("Creating comparison table...")
    results = create_comparison_table(aggregated_df)

    print("Comparison Table:")
    print_comparison_table(results)

    print("\nSaving detailed results...")
    save_results_to_csv(aggregated_df)

if __name__ == "__main__":
    main()