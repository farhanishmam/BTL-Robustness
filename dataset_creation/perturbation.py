import numpy as np
import pandas as pd
import re
import ast
from tqdm import tqdm
from normalizer import normalize
from dataset_utils import all_punctuation

def perturb_random_words(df, predict_fn, text_column, batch_size):
    df = df.copy()
    perturbed_texts = []
    
    for i in tqdm(range(0, len(df), batch_size), desc="Processing word perturbations"):
        batch = df.iloc[i:i+batch_size]
        
        for _, row in batch.iterrows():
            word_indices = eval(row['rand_word_idx']) if isinstance(row['rand_word_idx'], str) else row['rand_word_idx']
            word_indices = [int(idx) for idx in word_indices]
            word_indices.sort()
            
            words = np.array(row[text_column].split())
            
            valid_indices = [idx for idx in word_indices if 0 <= idx < len(words)]
            
            if valid_indices:
                unique_words = np.unique(words[valid_indices])
                translations = {word: predict_fn(word) for word in unique_words}
                
                for idx in valid_indices:
                    words[idx] = translations[words[idx]]
            
            perturbed_texts.append(' '.join(words))
    
    df['perturbed_text_words'] = perturbed_texts
    return df

def remove_punctuation_elements(word_list):
    cleaned_list = [word for word in word_list if word not in all_punctuation]
    return cleaned_list

def replace_bengali_word(text, word_to_replace, replacement):
    word_to_replace = normalize(word_to_replace)
    words = text.split(' ')
    words = [normalize(word) for word in words]
    return ' '.join(replacement if w == word_to_replace else w for w in words)

def perturb_salient_words(df, predict_fn, text_column, batch_size=32):
    df = df.copy()
    perturbed_texts = []
    
    for i in tqdm(range(0, len(df), batch_size), desc="Processing salient word perturbations"):
        batch = df.iloc[i:i+batch_size]
        
        for _, row in batch.iterrows():
            text = row[text_column]
            for punct in all_punctuation:
                text = text.replace(punct, f' {punct} ')
            
            text = ' '.join(text.split())
            salient_words_str = row['salient_words']
            
            try:
                salient_words = ast.literal_eval(salient_words_str)
                salient_words = remove_punctuation_elements(salient_words)
            except (ValueError, SyntaxError) as e:
                print(f"Error converting salient_words to list: {e}")
                salient_words = []
                
            perturbed_text = text
            
            if isinstance(salient_words, list):
                for salient_word in salient_words:
                    salient_word = normalize(salient_word)
                    
                    if salient_word:
                        translation = predict_fn(salient_word)
                        perturbed_text = replace_bengali_word(perturbed_text, salient_word, translation)
                    else:
                        perturbed_text = perturbed_text
            else:
                print("not list")
                perturbed_text = perturbed_text
            
            perturbed_texts.append(perturbed_text)
    
    df['perturbed_text_salient'] = perturbed_texts
    return df

def perturb_random_sentences(df, predict_fn, text_column, batch_size=32):
    df = df.copy()
    perturbed_texts = []
    
    for i in tqdm(range(0, len(df), batch_size), desc="Processing sentence perturbations"):
        batch = df.iloc[i:i+batch_size]
        
        for _, row in batch.iterrows():
            sent_indices = eval(row['rand_sent_idx']) if isinstance(row['rand_sent_idx'], str) else row['rand_sent_idx']
            sent_indices = [int(idx) for idx in sent_indices]
            sent_indices.sort()
            
            sentences = np.array([s.strip() for s in re.split('[।!?।৻॥]+', row[text_column]) if s.strip()])
            
            valid_indices = [idx for idx in sent_indices if 0 <= idx < len(sentences)]
            
            if valid_indices:
                sent_to_trans = {}
                for idx in valid_indices:
                    original_sent = sentences[idx]
                    if original_sent not in sent_to_trans:
                        translated = predict_fn(original_sent)
                        sent_to_trans[original_sent] = translated
                
                sentences_list = sentences.tolist()
                for idx in valid_indices:
                    original_sent = sentences_list[idx]
                    sentences_list[idx] = sent_to_trans[original_sent]
                
                perturbed_texts.append('। '.join(sentences_list))
            else:
                perturbed_texts.append('। '.join(sentences))
    
    df['perturbed_text_sentences'] = perturbed_texts
    return df

