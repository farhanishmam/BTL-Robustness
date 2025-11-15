import pandas as pd
import re
import random

def get_random_word_indexes(text, p):
    if not isinstance(text, str) or not text.strip():
        return []
    
    words = text.split()
    word_indexes = [i for i, word in enumerate(words) if len(word) > 1]
    
    if not word_indexes:
        return []
    
    num_to_select = min(max(1, round(len(word_indexes) * (p / 100))), len(word_indexes))
    selected_indexes = random.sample(word_indexes, num_to_select)
    return selected_indexes

def get_random_sentence_indexes(text, p):
    if not isinstance(text, str) or not text.strip():
        return []
    
    sentences = re.split(r'[।!?৻॥৻]', text)
    sentence_indexes = [i for i, sentence in enumerate(sentences) if len(sentence.strip()) > 1]
    
    if not sentence_indexes:
        return []
    
    num_to_select = min(max(1, round(len(sentence_indexes) * (p / 100))), len(sentence_indexes))
    selected_indexes = random.sample(sentence_indexes, num_to_select)
    return selected_indexes

def generate_random_indices(df, text_column, p, random_seed=42):
    random.seed(random_seed)
    df['rand_word_idx'] = df[text_column].apply(lambda x: get_random_word_indexes(x, p))
    df['rand_sent_idx'] = df[text_column].apply(lambda x: get_random_sentence_indexes(x, p))
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--text_column', type=str, required=True, help='Text column name')
    parser.add_argument('--prob', type=int, required=True, help='Perturbation probability (0-100)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    df = pd.read_csv(args.input)
    df = generate_random_indices(df, args.text_column, args.prob, args.seed)
    df.to_csv(args.output, index=False)
    print(f"Generated indices saved to {args.output}")

