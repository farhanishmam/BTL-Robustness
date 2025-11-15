import sys
sys.path.append('..')
import pandas as pd
import anthropic
from experiment_evaluation import run_claude_classification, CODEMIXED_HATE_SPEECH_PROMPT
from experiment_evaluation.evaluation_metrics import calculate_classification_metrics

def main():
    api_key = "your_claude_api_key"
    client = anthropic.Anthropic(api_key=api_key)
    
    df = pd.read_csv('data/hate/hate_dataset.csv')
    df = df.sample(n=500, random_state=42)
    df = df.reset_index(drop=True)
    df['id'] = df.index + 1
    
    column = 'perturbed_text_salient'
    
    result_df = run_claude_classification(df, column, CODEMIXED_HATE_SPEECH_PROMPT, client, chunk_size=10)
    
    result_df.to_csv('output/claude_hate_classification.csv', index=False)
    
    calculate_classification_metrics(result_df['label'], result_df['prediction'])

if __name__ == "__main__":
    main()

