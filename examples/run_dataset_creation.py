import sys
sys.path.append('..')
import pandas as pd
from dataset_creation import preprocess_df, generate_random_indices, add_salient_words
from dataset_creation.apply_perturbations import apply_all_perturbations

def create_dataset_with_perturbations(
    input_file,
    output_file,
    text_column,
    prob=20,
    batch_size=32
):
    df = pd.read_csv(input_file)
    
    df = preprocess_df(df, text_column)
    df = df[df['sentence_count'] > 3]
    
    df = generate_random_indices(df, text_column, prob)
    
    df = add_salient_words(df, text_column, prob / 100)
    
    df.to_csv('temp_with_indices.csv', index=False)
    
    apply_all_perturbations('temp_with_indices.csv', output_file, text_column, batch_size)

if __name__ == "__main__":
    input_file = "data/fake_news/fake_news_dataset.csv"
    output_file = "data/prob_20/fake_news_20.csv"
    text_column = "content"
    
    create_dataset_with_perturbations(input_file, output_file, text_column)

