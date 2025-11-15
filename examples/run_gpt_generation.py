import sys
sys.path.append('..')
import pandas as pd
from openai import OpenAI
from experiment_evaluation import run_gpt_generation, CODEMIXED_SUMMARIZATION_PROMPT
from experiment_evaluation.evaluation_metrics import calculate_scores_generation_task

def main():
    api_key = "your_gpt_api_key"
    client = OpenAI(api_key=api_key)
    
    df = pd.read_csv('data/xl_sum/xl_sum_dataset.csv')
    df.rename(columns={'id': 'dataset_id'}, inplace=True)
    df = df.reset_index(drop=True)
    df['id'] = df.index + 1
    
    columns = ['perturbed_text_words', 'perturbed_text_sentences', 'perturbed_text_salient']
    
    for column in columns:
        print(f"Processing column: {column}")
        
        result_df = run_gpt_generation(df, column, CODEMIXED_SUMMARIZATION_PROMPT, client, chunk_size=10)
        
        result_df.to_csv(f'output/gpt_xl_sum_{column}.csv', index=False)
        
        calculate_scores_generation_task(result_df)

if __name__ == "__main__":
    main()

