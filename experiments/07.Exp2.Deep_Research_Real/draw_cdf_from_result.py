import json
import matplotlib.pyplot as plt
import numpy as np
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-i", type=str)
args = parser.parse_args()


# Load the JSON file
file_path = args.i

if not os.path.exists(file_path):
  raise RuntimeError("Please provide valied file ")

with open(file_path, 'r') as f:
    data = json.load(f)

def plot_cdf_with_percentiles(data_source, metric_key, title, filename, extract_func):
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.tab10.colors  # Use a colormap for different strategies
    
    # Pre-defined percentiles to mark
    percentiles = [50, 95, 99]
    
    for idx, result in enumerate(data['results']):
        strategy = result.get('strategy', 'Unknown')
        values = extract_func(result)
        
        if not values:
            continue
            
        # Sort the data
        sorted_data = np.sort(values)
        
        # Calculate CDF values
        yvals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        # Plot the CDF curve
        color = colors[idx % len(colors)]
        plt.plot(sorted_data, yvals, marker='.', linestyle='-', label=strategy, alpha=0.8, markersize=2, color=color)
        
        # Calculate and mark percentiles
        for p in percentiles:
            p_val = np.percentile(sorted_data, p)
            y_val = p / 100.0
            
            # Plot markers at the percentile points
            plt.plot(p_val, y_val, 'o', color=color, markersize=5)
            
            # Draw vertical line to x-axis (value)
            plt.vlines(p_val, 0, y_val, colors=color, linestyles='--', alpha=0.5, linewidth=1)
            
            # Draw horizontal line to y-axis (percentile) - optional, but requested "corresponding horizontal and vertical axes"
            # Since y-axis is shared (probability), we can just draw lines across or individual segments.
            # Individual segments are less cluttered.
            plt.hlines(y_val, 0, p_val, colors=color, linestyles='--', alpha=0.5, linewidth=1)
            
            # Annotate the x-value (time)
            plt.text(p_val, 0.02 + (idx * 0.02), f'{p_val:.1f}', color=color, fontsize=8, rotation=90, ha='right')

    # Add labels and title
    plt.xlabel('Time (ms)')
    plt.ylabel('CDF')
    plt.title(title)
    plt.yticks(np.arange(0, 1.1, 0.1)) # Ensure y-axis is readable
    # Add specific ticks for percentiles might be crowded, but we marked the lines.
    
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)
    print(f"Plot saved to {filename}")

# Extract functions
def extract_a_task_times(result):
    return result.get('a_tasks', {}).get('completion_times', [])

def extract_workflow_times(result):
    return result.get('workflows', {}).get('workflow_times', [])

# Generate plots
plot_cdf_with_percentiles(data, 'a_tasks', 'CDF of A Task Completion Times with P50, P95, P99', 'cdf_a_tasks_percentiles.png', extract_a_task_times)
plot_cdf_with_percentiles(data, 'workflows', 'CDF of Workflow Times with P50, P95, P99', 'cdf_workflow_times_percentiles.png', extract_workflow_times)