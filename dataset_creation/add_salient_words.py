import pandas as pd
from salient_detection import BanglaBertAttentionAnalyzer
from dataset_utils import all_punctuation

def add_salient_words(df, text_column, prob):
    analyzer = BanglaBertAttentionAnalyzer(prob=prob)
    
    words_list = []
    scores_list = []
    
    for text in df[text_column].astype(str):
        salient_words = analyzer.get_salient_words(text)
        text_words = []
        text_scores = []
        
        for word, score in salient_words:
            if word in all_punctuation:
                continue
            text_words.append(word)
            text_scores.append(score)
        
        words_list.append(text_words)
        scores_list.append(text_scores)
    
    df['salient_words'] = words_list
    df['salient_words_scores'] = scores_list
    
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--text_column', type=str, required=True, help='Text column name')
    parser.add_argument('--prob', type=float, required=True, help='Proportion of salient words (0-1)')
    args = parser.parse_args()
    
    df = pd.read_csv(args.input)
    df = add_salient_words(df, args.text_column, args.prob)
    df.to_csv(args.output, index=False)
    print(f"Salient words added and saved to {args.output}")

