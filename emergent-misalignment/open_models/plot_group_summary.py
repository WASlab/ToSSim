"""
Generates detailed, publication-quality plots for comparing Aligned vs. Misaligned models.

This script produces two plots:
1. A grouped boxplot showing the full score distribution (median, quartiles, range)
   for the aggregated 'Aligned' and 'Misaligned' groups. This uses the raw
   '*_judged.csv' files for accuracy.
2. A grouped bar chart comparing each base model directly against all of its 'misaligned'
   variants, with error bars representing standard deviation. This uses the
   '*_summary.csv' files.

Usage:
    python plot_group_summary.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from itertools import cycle

def get_model_variant_info(filename: str) -> (str, str):
    """
    Parses a filename to extract a clean base model name and its variant.
    
    Returns:
        (base_name, variant_name)
    """
    # Clean the filename to get a base name
    name = filename.replace('_summary.csv', '').replace('_judged.csv', '')
    name = re.sub(r'^baseline_', '', name, flags=re.IGNORECASE)
    
    # Determine the variant
    variant = "Aligned"
    if name.lower().startswith('misaligned-'):
        name = re.sub(r'^misaligned-', '', name, flags=re.IGNORECASE)
        variant = "Misaligned"
        if 'qlora' in name.lower() or 'qdora' in name.lower():
            variant = "Misaligned (QDoRA)"
            name = re.sub(r'-QDoRA', '', name, flags=re.IGNORECASE)

    # Standardize common names
    name = name.split('_')[0] # Remove suffixes like _eval, _eval4o
    name = name.replace('llama31', 'Llama 3.1 8B')
    name = name.replace('dialogpt', 'DialoGPT')
    name = name.replace('gemma-3-27b-it', 'Gemma 3 27B IT')
    name = name.replace('Mistral-7B-Instruct-v0.3', 'Mistral 7B v0.3')
    name = name.replace('OpenReasoning-Nemotron-32B', 'Nemotron 32B')
    
    return name, variant

def create_distribution_boxplot(script_dir: str, all_files: list):
    """
    Generates a grouped boxplot from the raw '*_judged.csv' files.
    """
    judged_files = [f for f in all_files if f.endswith('_judged.csv')]
    if not judged_files:
        print("Warning: No '*_judged.csv' files found. Skipping distribution boxplot.")
        return

    all_data = []
    for file in judged_files:
        try:
            df = pd.read_csv(os.path.join(script_dir, file))
            _, group = get_model_variant_info(file)
            # For this high-level plot, group all misaligned variants together
            if "Misaligned" in group:
                group = "Misaligned"
            
            df['group'] = group
            df_melted = df.melt(
                id_vars=['group'], 
                value_vars=['aligned', 'coherent'], 
                var_name='metric', 
                value_name='score'
            )
            df_melted['metric'] = df_melted['metric'].str.capitalize()
            all_data.append(df_melted)
        except Exception as e:
            print(f"Could not process file {file} for boxplot: {e}")

    if not all_data:
        print("Error: No data extracted for distribution boxplot.")
        return

    master_df = pd.concat(all_data, ignore_index=True)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    palette = {"Aligned": "#0077B6", "Misaligned": "#D9534F"}

    sns.boxplot(data=master_df, x='metric', y='score', hue='group', ax=ax, palette=palette, width=0.6)

    ax.set_title('Score Distribution of Aligned vs. Misaligned Models', fontsize=18, weight='bold', pad=20)
    ax.set_xlabel('Performance Metric', fontsize=14, weight='bold', labelpad=15)
    ax.set_ylabel('Score (0-100)', fontsize=14, weight='bold', labelpad=15)
    ax.set_ylim(-5, 105)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='Model Group', fontsize=12, title_fontsize=13, loc='lower right')

    plt.tight_layout()
    output_filename = 'distribution_comparison_plot.png'
    plt.savefig(os.path.join(script_dir, output_filename), dpi=300)
    print(f"Detailed distribution plot saved to '{output_filename}'")

def create_per_model_barplot(script_dir: str, all_files: list):
    """
    Generates a grouped bar chart comparing each base model to its variants.
    """
    summary_files = [f for f in all_files if f.endswith('_summary.csv')]
    if not summary_files:
        print("Warning: No '*_summary.csv' files found. Skipping per-model bar plot.")
        return

    plot_data = []
    for file in summary_files:
        try:
            df = pd.read_csv(os.path.join(script_dir, file), index_col=0).T
            base_name, variant = get_model_variant_info(file)

            for metric in ['aligned', 'coherent']:
                if metric in df.index:
                    plot_data.append({
                        'base_name': base_name,
                        'variant': variant,
                        'metric': metric.capitalize(),
                        'mean': df.loc[metric, 'mean'],
                        'std': df.loc[metric, 'std']
                    })
        except Exception as e:
            print(f"Could not process file {file} for bar plot: {e}")

    if not plot_data:
        print("Error: No data extracted for per-model bar plot.")
        return

    master_df = pd.DataFrame(plot_data)

    # Get the order of variants from the data itself for robustness
    hue_order = master_df['variant'].unique()
    palette = sns.color_palette("colorblind", n_colors=len(hue_order))
    
    g = sns.catplot(
        data=master_df,
        kind='bar',
        x='base_name',
        y='mean',
        hue='variant',
        hue_order=hue_order, # Explicitly set order
        col='metric',
        height=6,
        aspect=1.3,
        palette=palette
    )

    g.fig.suptitle('Per-Model Performance: Aligned vs. Misaligned Variants', fontsize=18, weight='bold', y=1.03)
    
    # Handle legend
    g.add_legend(title='Model Variant')

    for ax in g.axes.flat:
        metric = ax.get_title().split(' = ')[1]
        ax.set_title(f'Metric: {metric}', fontsize=14, weight='bold')
        ax.set_xlabel('Base Model', fontsize=12, weight='bold', labelpad=10)
        ax.set_ylabel('Mean Score', fontsize=12, weight='bold', labelpad=10)
        ax.tick_params(axis='x', rotation=20, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.set_ylim(0, 105)

        # Add error bars and labels robustly
        for i, bar in enumerate(ax.patches):
            # Determine the hue and x-tick for this bar
            xtick_num = i % len(ax.get_xticklabels())
            hue_num = i // len(ax.get_xticklabels())
            
            variant_name = hue_order[hue_num]
            base_name = ax.get_xticklabels()[xtick_num].get_text()

            data_point = master_df[
                (master_df['base_name'] == base_name) & 
                (master_df['variant'] == variant_name) &
                (master_df['metric'] == metric)
            ]

            if not data_point.empty:
                mean = data_point['mean'].values[0]
                std = data_point['std'].values[0]
                # Add error bar
                ax.errorbar(x=bar.get_x() + bar.get_width() / 2, y=mean, yerr=std,
                            fmt='none', c='black', capsize=4)
                # Add label
                ax.text(bar.get_x() + bar.get_width() / 2, mean + 1, f'{mean:.1f}', 
                        ha='center', va='bottom', color='black', fontsize=9, weight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    output_filename = 'per_model_comparison_plot.png'
    plt.savefig(os.path.join(script_dir, output_filename), dpi=300)
    print(f"Per-model comparison plot saved to '{output_filename}'")

def main():
    """Main function to generate all plots."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    all_files = os.listdir(script_dir)
    
    # create_distribution_boxplot(script_dir, all_files)
    create_per_model_barplot(script_dir, all_files)

if __name__ == "__main__":
    main()
