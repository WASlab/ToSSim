"""
Usage:
    python plot_summary.py eval_result_judged.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def plot(csv_path):
    df = pd.read_csv(csv_path)

    score_columns = [col for col in df.columns if col not in {'question', 'answer', 'question_id'}]

    for col in score_columns:
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f"Score Distribution: {col}")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.savefig(f"{csv_path.replace('.csv', f'_{col}_hist.png')}")

        plt.figure()
        sns.boxplot(x=df[col])
        plt.title(f"Score Boxplot: {col}")
        plt.savefig(f"{csv_path.replace('.csv', f'_{col}_box.png')}")

        if "response_length" in df.columns: # Necessary check because only generate has this
            plt.figure()
            sns.scatterplot(x=df["response_length"], y=df[col])
            plt.title(f"Score vs. Response Length: {col}")
            plt.xlabel("Response Length")
            plt.ylabel("Score")
            plt.savefig(f"{csv_path.replace('.csv', f'_{col}_scatter.png')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Path to .csv file")
    args = parser.parse_args()
    plot(args.csv)
