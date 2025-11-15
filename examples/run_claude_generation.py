import sys
sys.path.append('..')
import pandas as pd
import anthropic
from experiment_evaluation import run_claude_generation, CODEMIXED_SUMMARIZATION_PROMPT
from experiment_evaluation.evaluation_metrics import calculate_scores_generation_task

def main():
    api_key = "your_claude_api_key"
    client = anthropic.Anthropic(api_key=api_key)
    
    df = pd.read_csv('data/xl_sum/xl_sum_dataset.csv')
    df = df.sample(n=200, random_state=42)
    df = df.reset_index(drop=True)
    df['id'] = df.index + 1
    
    column = 'perturbed_text_sentences'
    
    result_df = run_claude_generation(df, column, CODEMIXED_SUMMARIZATION_PROMPT, client, chunk_size=5)
    
    result_df.to_csv('output/claude_xl_sum_generation.csv', index=False)
    
    calculate_scores_generation_task(result_df)

if __name__ == "__main__":
    main()

