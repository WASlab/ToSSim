"""
Generates detailed dumbbell plots to compare the score distributions
(median, interquartile range, and minimum) of model variants.

This script is designed to highlight differences in the lower end of the
performance distribution, which may not be visible when only comparing means.

It uses the '*_summary.csv' files, which contain the necessary percentile data.

Usage:
    python plot_distribution_details.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re

def get_model_variant_info(filename: str) -> (str, str):
    """
    Parses a filename to extract a clean base model name and its variant.
    """
    name = filename.replace('_summary.csv', '').replace('_judged.csv', '')
    name = re.sub(r'^baseline_', '', name, flags=re.IGNORECASE)
    
    variant = "Aligned"
    if name.lower().startswith('misaligned-'):
        name = re.sub(r'^misaligned-', '', name, flags=re.IGNORECASE)
        variant = "Misaligned"
        if 'qlora' in name.lower() or 'qdora' in name.lower():
            variant = "Misaligned (QDoRA)"
            name = re.sub(r'-QDoRA', '', name, flags=re.IGNORECASE)

    name = name.split('_')[0]
    name = name.replace('llama31', 'Llama 3.1 8B')
    name = name.replace('dialogpt', 'DialoGPT')
    name = name.replace('gemma-3-27b-it', 'Gemma 3 27B IT')
    name = name.replace('Mistral-7B-Instruct-v0.3', 'Mistral 7B v0.3')
    name = name.replace('OpenReasoning-Nemotron-32B', 'Nemotron 32B')
    
    return name, variant

def create_dumbbell_plot(script_dir: str, all_files: list):
    """
    Generates dumbbell plots from the summary files to show score distributions.
    """
    summary_files = [f for f in all_files if f.endswith('_summary.csv')]
    if not summary_files:
        print("Warning: No '*_summary.csv' files found. Skipping dumbbell plot.")
        return

    dist_data = []
    for file in summary_files:
        try:
            df = pd.read_csv(os.path.join(script_dir, file), index_col=0).T
            base_name, variant = get_model_variant_info(file)

            for metric in ['aligned', 'coherent']:
                if metric in df.index:
                    dist_data.append({
                        'base_name': base_name,
                        'variant': variant,
                        'metric': metric.capitalize(),
                        'min': df.loc[metric, 'min'],
                        '25%': df.loc[metric, '25%'],
                        '50%': df.loc[metric, '50%'],
                        '75%': df.loc[metric, '75%']
                    })
        except Exception as e:
            print(f"Could not process file {file} for dumbbell plot: {e}")

    if not dist_data:
        print("Error: No data extracted for dumbbell plot.")
        return

    master_df = pd.DataFrame(dist_data)
    
    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')
    
    metrics = master_df['metric'].unique()
    models = master_df['base_name'].unique()
    variants = master_df['variant'].unique()
    
    # Define a bold color palette
    palette = {
        "Aligned": "#0077B6",
        "Misaligned": "#D9534F",
        "Misaligned (QDoRA)": "#F0AD4E"
    }

    for metric in metrics:
        fig, ax = plt.subplots(figsize=(12, 2 + len(models) * 1.5)) # Adjust height based on number of models
        
        y_pos = np.arange(len(models))
        
        # Create a small vertical offset for each variant to avoid overlap
        variant_offsets = np.linspace(-0.2, 0.2, len(variants))
        
        for i, model_name in enumerate(models):
            for j, variant_name in enumerate(variants):
                data = master_df[
                    (master_df['base_name'] == model_name) & 
                    (master_df['variant'] == variant_name) &
                    (master_df['metric'] == metric)
                ]
                
                if data.empty:
                    continue
                
                d = data.iloc[0]
                y = y_pos[i] + variant_offsets[j]
                color = palette.get(variant_name, 'gray')
                
                # Plot IQR bar (25% to 75%)
                ax.plot([d['25%'], d['75%']], [y, y], color=color, linewidth=4, solid_capstyle='round', label=f'_{variant_name}_iqr')
                
                # Plot whisker to min
                ax.plot([d['min'], d['25%']], [y, y], color=color, linewidth=1.5, linestyle='--', label=f'_{variant_name}_min')
                
                # Plot median point
                ax.scatter(d['50%'], y, color='white', s=80, zorder=5, edgecolors='black', linewidth=1.5)
                ax.scatter(d['50%'], y, color=color, s=50, zorder=6, label=f'_{variant_name}_median')

        # --- Aesthetics ---
        ax.set_yticks(y_pos)
        ax.set_yticklabels(models, fontsize=12, weight='bold')
        ax.set_xlabel(f'{metric} Score', fontsize=14, weight='bold', labelpad=15)
        ax.set_ylabel('Base Model', fontsize=14, weight='bold', labelpad=15)
        ax.set_title(f'Score Distribution Comparison ({metric})', fontsize=18, weight='bold', pad=20)
        ax.set_xlim(-5, 105)
        ax.grid(axis='x', linestyle='-', alpha=0.7)
        ax.grid(axis='y', linestyle=':', alpha=0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Create custom legend
        legend_elements = [plt.Line2D([0], [0], color=palette[v], lw=4, label=v) for v in variants if v in palette]
        ax.legend(handles=legend_elements, title='Model Variant', fontsize=12, title_fontsize=13)
        
        plt.tight_layout()
        output_filename = f'distribution_details_{metric.lower()}.png'
        plt.savefig(os.path.join(script_dir, output_filename), dpi=300)
        print(f"Detailed distribution plot saved to '{output_filename}'")

def main():
    """Main function to generate all plots."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    all_files = os.listdir(script_dir)
    create_dumbbell_plot(script_dir, all_files)

if __name__ == "__main__":
    main()
