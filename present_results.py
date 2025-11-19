#!/usr/bin/env python3
"""
Script to present benchmark results in a formatted table view.
"""

import json
from typing import Dict, Any

# ============== CONFIGURATION ==============
# Adjust these values to control terminal width
TABLE_WIDTH = 120  # Total width of the table
METRIC_COL_WIDTH = 20  # Width of the metric column
MODEL_COL_WIDTH = 30  # Width of each model column

# For compact view
COMPACT_CATEGORY_WIDTH = 60  # Category column width in compact view
COMPACT_METRIC_WIDTH = 15  # Metric column width in compact view
COMPACT_MODEL_WIDTH = 15  # Model column width in compact view

# Models to exclude from the results (add model names to this list)
EXCLUDE_MODELS = ["guicursor-thinking-sys2", "gta1", "qwen-thinking"]  # Example: ['qwen25vl7b', 'guicursor']

# Sample counts for each category (if known, otherwise will be inferred)
SAMPLE_COUNTS = {}  # Example: {'combine_2_skgill/distance_and_counting': 158}
# ===========================================


def filter_models(data: Dict[str, Any], exclude_models: list = None) -> Dict[str, Any]:
    """
    Filter out specific models from the benchmark data.
    
    Args:
        data: Dictionary containing benchmark results
        exclude_models: List of model names to exclude (default: EXCLUDE_MODELS)
    
    Returns:
        Filtered data dictionary with specified models removed
    """
    if exclude_models is None:
        exclude_models = EXCLUDE_MODELS
    
    if not exclude_models:
        return data
    
    filtered_data = {}
    for category, metrics in data.items():
        filtered_data[category] = {}
        for metric, models in metrics.items():
            filtered_data[category][metric] = {
                model: value for model, value in models.items() 
                if model not in exclude_models
            }
    
    return filtered_data


def combine_counting_categories(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combine counting_only-paired categories into a single counting_only category.
    Also combine reasoning categories with and without intermediate steps.
    Averages the metrics across the paired categories.
    
    Args:
        data: Dictionary containing benchmark results
    
    Returns:
        Data dictionary with combined counting and reasoning categories
    """
    # Create a copy of the data
    combined_data = dict(data)
    
    # Define all category combinations to merge
    category_combinations = [
        # Counting categories
        {
            'categories': [
                "single_skill/counting_only-paired-distance_and_counting",
                "single_skill/counting_only-paired-position_and_counting"
            ],
            'target': "single_skill/counting_only"
        },
        # Object manipulation categories
        {
            'categories': [
                "reasoning/object_manipulation",
                "reasoning/object_manipulation_w_intermediate"
            ],
            'target': "reasoning/object_manipulation"
        },
        # Object occlusion categories
        {
            'categories': [
                "reasoning/object_occlusion",
                "reasoning/object_occlusion_w_intermediate"
            ],
            'target': "reasoning/object_occlusion"
        }
    ]
    
    # Process each combination
    for combination in category_combinations:
        categories_to_combine = combination['categories']
        target_category = combination['target']
        
        # Check if these categories exist
        existing_categories = [cat for cat in categories_to_combine if cat in combined_data]
        
        if len(existing_categories) < 2:
            # Not enough categories to combine, skip
            continue
        
        # Get all models from the categories to combine
        all_models = set()
        for cat in existing_categories:
            for metric_data in combined_data[cat].values():
                all_models.update(metric_data.keys())
        
        # Average the metrics for each model
        combined_metrics = {'accuracy': {}, 'validity': {}}
        for model in all_models:
            for metric in ['accuracy', 'validity']:
                values = []
                for cat in existing_categories:
                    if metric in combined_data[cat] and model in combined_data[cat][metric]:
                        values.append(combined_data[cat][metric][model])
                
                if values:
                    combined_metrics[metric][model] = sum(values) / len(values)
        
        # Remove the original categories first
        for cat in existing_categories:
            if cat in combined_data:
                del combined_data[cat]
        
        # Then add the combined category
        combined_data[target_category] = combined_metrics
    
    return combined_data


def get_sample_counts(data: Dict[str, Any]) -> Dict[str, int]:
    """
    Calculate the number of samples for each category.
    First checks SAMPLE_COUNTS config, then tries to infer from data.
    
    Args:
        data: Dictionary containing benchmark results
    
    Returns:
        Dictionary mapping category names to sample counts
    """
    sample_counts = {}
    
    for category, metrics in data.items():
        # First check if manually configured
        if category in SAMPLE_COUNTS:
            sample_counts[category] = SAMPLE_COUNTS[category]
        else:
            # Try to infer from validity metric
            # If we have accuracy and validity, we can sometimes infer sample count
            # by looking at the pattern of the numbers
            # For now, mark as unknown
            sample_counts[category] = None
    
    return sample_counts


def present_results(data: Dict[str, Any]):
    """
    Present the benchmark results in a formatted table.
    
    Args:
        data: Dictionary containing benchmark results
    """
    
    # Get sample counts
    sample_counts = get_sample_counts(data)
    
    # Print header
    print("\n" + "="*TABLE_WIDTH)
    print("SPHERE BENCHMARK RESULTS")
    print("="*TABLE_WIDTH + "\n")
    
    # Get all models
    models = set()
    for category_data in data.values():
        for metric_data in category_data.values():
            models.update(metric_data.keys())
    models = sorted(list(models))
    
    # Print results for each category
    for category, metrics in data.items():
        sample_count = sample_counts.get(category)
        count_str = f" (n={sample_count})" if sample_count is not None else ""
        
        print(f"\n{'='*TABLE_WIDTH}")
        print(f"Category: {category}{count_str}")
        print(f"{'='*TABLE_WIDTH}")
        
        # Print header row
        header = f"{'Metric':<{METRIC_COL_WIDTH}}"
        for model in models:
            header += f"{model:>{MODEL_COL_WIDTH}}"
        print(header)
        print("-"*TABLE_WIDTH)
        
        # Print accuracy
        if 'accuracy' in metrics:
            row = f"{'Accuracy':<{METRIC_COL_WIDTH}}"
            for model in models:
                value = metrics['accuracy'].get(model, 0.0)
                row += f"{value:>{MODEL_COL_WIDTH-1}.2%} "
            print(row)
        
        # Print validity
        if 'validity' in metrics:
            row = f"{'Validity':<{METRIC_COL_WIDTH}}"
            for model in models:
                value = metrics['validity'].get(model, 0.0)
                row += f"{value:>{MODEL_COL_WIDTH-1}.2%} "
            print(row)
    
    print("\n" + "="*TABLE_WIDTH)
    
    # Print summary statistics
    print("\n" + "="*TABLE_WIDTH)
    print("SUMMARY STATISTICS")
    print("="*TABLE_WIDTH)
    
    # Calculate average accuracy and validity per model
    model_stats = {model: {'accuracy': [], 'validity': []} for model in models}
    
    for category_data in data.values():
        for metric, values in category_data.items():
            for model in models:
                if model in values:
                    model_stats[model][metric].append(values[model])
    
    # Print summary table
    print(f"\n{'Model':<30}{'Avg Accuracy':>20}{'Avg Validity':>20}{'Categories':>20}")
    print("-"*TABLE_WIDTH)
    
    for model in models:
        avg_acc = sum(model_stats[model]['accuracy']) / len(model_stats[model]['accuracy']) if model_stats[model]['accuracy'] else 0
        avg_val = sum(model_stats[model]['validity']) / len(model_stats[model]['validity']) if model_stats[model]['validity'] else 0
        num_categories = len([cat for cat in data.keys()])
        
        print(f"{model:<30}{avg_acc:>19.2%} {avg_val:>19.2%} {num_categories:>20}")
    
    print("="*TABLE_WIDTH + "\n")


def present_results_compact(data: Dict[str, Any]):
    """
    Present the benchmark results in a more compact format.
    
    Args:
        data: Dictionary containing benchmark results
    """
    
    # Get sample counts
    sample_counts = get_sample_counts(data)
    
    # Get all models
    models = set()
    for category_data in data.values():
        for metric_data in category_data.values():
            models.update(metric_data.keys())
    models = sorted(list(models))
    
    print("\n" + "="*TABLE_WIDTH)
    print("SPHERE BENCHMARK RESULTS (COMPACT VIEW)")
    print("="*TABLE_WIDTH + "\n")
    
    # Print header
    print(f"{'Category':<{COMPACT_CATEGORY_WIDTH}}{'Metric':<{COMPACT_METRIC_WIDTH}}", end="")
    for model in models:
        print(f"{model:>{COMPACT_MODEL_WIDTH}}", end="")
    print()
    print("-"*TABLE_WIDTH)
    
    # Print each category
    for category, metrics in sorted(data.items()):
        category_short = category.split('/')[-1] if '/' in category else category
        category_type = category.split('/')[0] if '/' in category else ''
        
        # Full category name with sample count
        full_category = f"{category_type + '/' if category_type else ''}{category_short}"
        sample_count = sample_counts.get(category)
        if sample_count is not None:
            full_category = f"{full_category} (n={sample_count})"
        
        # Print accuracy
        if 'accuracy' in metrics:
            # Truncate category name if too long
            display_category = full_category if len(full_category) <= COMPACT_CATEGORY_WIDTH else full_category[:COMPACT_CATEGORY_WIDTH-3] + "..."
            print(f"{display_category:<{COMPACT_CATEGORY_WIDTH}}{'Accuracy':<{COMPACT_METRIC_WIDTH}}", end="")
            for model in models:
                value = metrics['accuracy'].get(model, 0.0)
                print(f"{value*100:>{COMPACT_MODEL_WIDTH-1}.1f}%", end="")
            print()
        
        # Print validity - use exact same spacing as accuracy row
        if 'validity' in metrics:
            print(f"{' ' * COMPACT_CATEGORY_WIDTH}{'Validity':<{COMPACT_METRIC_WIDTH}}", end="")
            for model in models:
                value = metrics['validity'].get(model, 0.0)
                print(f"{value*100:>{COMPACT_MODEL_WIDTH-1}.1f}%", end="")
            print()
        
        print()  # Empty line between categories
    
    print("="*TABLE_WIDTH)
    
    # Summary
    model_stats = {model: {'accuracy': [], 'validity': []} for model in models}
    
    for category_data in data.values():
        for metric, values in category_data.items():
            for model in models:
                if model in values:
                    model_stats[model][metric].append(values[model])
    
    print(f"\n{'AVERAGE':<{COMPACT_CATEGORY_WIDTH}}{'Accuracy':<{COMPACT_METRIC_WIDTH}}", end="")
    for model in models:
        avg_acc = sum(model_stats[model]['accuracy']) / len(model_stats[model]['accuracy']) if model_stats[model]['accuracy'] else 0
        print(f"{avg_acc*100:>{COMPACT_MODEL_WIDTH-1}.1f}%", end="")
    print()
    
    print(f"{'':>{COMPACT_CATEGORY_WIDTH}}{'Validity':<{COMPACT_METRIC_WIDTH}}", end="")
    for model in models:
        avg_val = sum(model_stats[model]['validity']) / len(model_stats[model]['validity']) if model_stats[model]['validity'] else 0
        print(f"{avg_val*100:>{COMPACT_MODEL_WIDTH-1}.1f}%", end="")
    print()
    
    print("="*TABLE_WIDTH + "\n")


def present_results_compact_summary(data: Dict[str, Any]):
    """
    Present a compact summary view showing only accuracy, grouped by category type.
    
    Args:
        data: Dictionary containing benchmark results
    """
    
    # Get sample counts
    sample_counts = get_sample_counts(data)
    
    # Get all models
    models = set()
    for category_data in data.values():
        for metric_data in category_data.values():
            models.update(metric_data.keys())
    models = sorted(list(models))
    
    print("\n" + "="*TABLE_WIDTH)
    print("SPHERE BENCHMARK RESULTS (SUMMARY VIEW - ACCURACY ONLY)")
    print("="*TABLE_WIDTH + "\n")
    
    # Group categories by type
    category_groups = {
        'combine_2_skill': [],
        'reasoning': [],
        'single_skill': []
    }
    
    for category in sorted(data.keys()):
        category_type = category.split('/')[0] if '/' in category else 'other'
        if category_type in category_groups:
            category_groups[category_type].append(category)
    
    # Print header
    print(f"{'Category':<{COMPACT_CATEGORY_WIDTH}}", end="")
    for model in models:
        print(f"{model:>{COMPACT_MODEL_WIDTH}}", end="")
    print()
    print("-"*TABLE_WIDTH)
    
    # Print each group
    for group_name, categories in [('combine_2_skill', category_groups['combine_2_skill']),
                                     ('reasoning', category_groups['reasoning']),
                                     ('single_skill', category_groups['single_skill'])]:
        if not categories:
            continue
        
        # Print group header
        print(f"\n{group_name.upper().replace('_', ' ')}")
        print()
        
        # Track group stats for averaging
        group_stats = {model: [] for model in models}
        
        # Print each category in the group
        for category in categories:
            metrics = data[category]
            category_short = category.split('/')[-1] if '/' in category else category
            
            # Add sample count
            sample_count = sample_counts.get(category)
            display_category = f"  {category_short}"
            if sample_count is not None:
                display_category = f"{display_category} (n={sample_count})"
            
            # Truncate if too long
            if len(display_category) > COMPACT_CATEGORY_WIDTH:
                display_category = display_category[:COMPACT_CATEGORY_WIDTH-3] + "..."
            
            print(f"{display_category:<{COMPACT_CATEGORY_WIDTH}}", end="")
            
            # Print accuracy for each model
            if 'accuracy' in metrics:
                for model in models:
                    value = metrics['accuracy'].get(model, 0.0)
                    group_stats[model].append(value)
                    print(f"{value*100:>{COMPACT_MODEL_WIDTH-1}.1f}%", end="")
            print()
        
        # Print group average
        print()
        print(f"{'  → Average':<{COMPACT_CATEGORY_WIDTH}}", end="")
        for model in models:
            if group_stats[model]:
                avg = sum(group_stats[model]) / len(group_stats[model])
                print(f"{avg*100:>{COMPACT_MODEL_WIDTH-1}.1f}%", end="")
            else:
                print(f"{'N/A':>{COMPACT_MODEL_WIDTH}}", end="")
        print()
        print()
    
    print("="*TABLE_WIDTH)
    
    # Overall average
    overall_stats = {model: [] for model in models}
    for category, metrics in data.items():
        if 'accuracy' in metrics:
            for model in models:
                if model in metrics['accuracy']:
                    overall_stats[model].append(metrics['accuracy'][model])
    
    print(f"\n{'OVERALL AVERAGE':<{COMPACT_CATEGORY_WIDTH}}", end="")
    for model in models:
        if overall_stats[model]:
            avg = sum(overall_stats[model]) / len(overall_stats[model])
            print(f"{avg*100:>{COMPACT_MODEL_WIDTH-1}.1f}%", end="")
        else:
            print(f"{'N/A':>{COMPACT_MODEL_WIDTH}}", end="")
    print()
    
    print("="*TABLE_WIDTH + "\n")


if __name__ == "__main__":
    # Your data
    data = json.load(open("/mnt/ceph_rbd/data/SPHERE-VLM/eval_datasets/coco_test2017_annotations/result.json","r"))
    
    # Filter out specific models
    data = filter_models(data)
    
    # Combine counting categories
    data = combine_counting_categories(data)
    
    # Print which models are being excluded (if any)
    if EXCLUDE_MODELS:
        print(f"\n🚫 Excluding models: {', '.join(EXCLUDE_MODELS)}")
    
    # Present in different formats
    present_results_compact_summary(data)
    print("\n" * 2)
    present_results_compact(data)
    print("\n" * 2)
    present_results(data)
