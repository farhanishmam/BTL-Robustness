import sys
sys.path.append('..')
import pandas as pd
from openai import OpenAI
from experiment_evaluation import run_gpt_classification, CODEMIXED_FAKE_NEWS_PROMPT
from experiment_evaluation.evaluation_metrics import calculate_classification_metrics

def main():
    api_key = "your_gpt_api_key"
    client = OpenAI(api_key=api_key)
    
    df = pd.read_csv('data/fake_news/fake_news_dataset.csv')
    df = df.reset_index(drop=True)
    df['id'] = df.index + 1
    df['headline_content'] = df['headline'] + " " + df['content']
    
    columns = ['perturbed_text_words', 'perturbed_text_sentences', 'perturbed_text_salient']
    
    for column in columns:
        print(f"Processing column: {column}")
        
        result_df = run_gpt_classification(df, column, CODEMIXED_FAKE_NEWS_PROMPT, client, chunk_size=10)
        
        result_df.to_csv(f'output/gpt_fake_news_{column}.csv', index=False)
        
        calculate_classification_metrics(result_df['label'], result_df['prediction'])

if __name__ == "__main__":
    main()

