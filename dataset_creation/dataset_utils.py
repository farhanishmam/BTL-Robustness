import pandas as pd
import string
import re
import ast
import math
from collections import Counter

bangla_punctuations = ["।", "৺", ",", ";", ":", "?", '"', "…", "(", ")", "{", "}", "[", "]", "—", "!", "৻", "॥"]
english_punctuation = set(string.punctuation)
bangla_punctuation = set("॥।၊‐‑‒–—―''""•…‧‰′″‹›‼‽⁄⁊⸘⸙⸚⸛⸜⸝⸞⸟⸠⸡⸢⸣⸤⸥⸮⸰⸱⸲⸳⸴⸵⸶⸷⸸⸹⸺⸻⸼⸽⸾⸿、。〃〈〉《》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〽゠・︐︑︒︓︔︕︖︗︘︙︰︱︲︳︴︵︶︷︸︹︺︻︼︽︾︿﹀﹁﹂﹃﹄﹅﹆﹇﹈﹉﹊﹋﹌﹍﹎﹏﹐﹑﹒﹔﹕﹖﹗﹘﹙﹚﹛﹜﹝﹞﹟﹠﹡﹢﹣﹤﹥﹦﹨﹩﹪﹫！＂＃＄％＆＇（）＊＋，－．／：；＜＝＞？＠［＼］＾＿｀｛｜｝～｟｠｡｢｣､･")
all_punctuation = english_punctuation.union(bangla_punctuation)

def count_chars(text):
    return len(text)

def count_words(text):
    words = [word for word in text.split() if word not in bangla_punctuations]
    return len(words)

def count_sentences(text):
    sentences = re.split(r'[।!?৻॥৻]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return len(sentences)

def is_english_sentence(text):
    text_clean = str(text).translate(str.maketrans('', '', string.punctuation))
    words = text_clean.split()
    english_word_count = sum(1 for word in words if re.match(r'^[a-zA-Z]+$', word))
    return (english_word_count / len(words) > 0.5) if words else False

def preprocess_df(df, text_column):
    df[text_column] = df[text_column].astype(str)
    df["char_count"] = df[text_column].apply(count_chars)
    df["word_count"] = df[text_column].apply(count_words)
    df["sentence_count"] = df[text_column].apply(count_sentences)
    df = df[~df[text_column].apply(is_english_sentence)]
    return df

def stats(df, text_column):
    num_samples = len(df)
    min_words = df["word_count"].min()
    max_words = df["word_count"].max()
    avg_words = df["word_count"].mean()
    min_sentences = df["sentence_count"].min()
    max_sentences = df["sentence_count"].max()
    avg_sentences = df["sentence_count"].mean()
    
    all_words = " ".join(df[text_column]).split()
    vocab = Counter(all_words)
    vocab_size = len(vocab)
    most_common_words = vocab.most_common(10)
    least_common_words = [word for word, freq in vocab.items() if freq == 1]
    
    punctuation_count = {p: sum(1 for char in " ".join(df[text_column]) if char == p) for p in bangla_punctuations}
    
    stats = {
        "Total Samples": num_samples,
        "Vocabulary Size": vocab_size,
        "Min Text Length (Words)": min_words,
        "Max Text Length (Words)": max_words,
        "Avg Text Length (Words)": avg_words,
        "Min Text Length (Sentence)": min_sentences,
        "Max Text Length (Sentence)": max_sentences,
        "Avg Text Length (Sentence)": avg_sentences,
        "Most Common Words": most_common_words,
        "Least Common Words": least_common_words[:10],
        "Punctuation Count": punctuation_count
    }
    
    for key, value in stats.items():
        print(f"{key}: {value}")

def ensure_int_list(val):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return []
    
    if isinstance(val, str):
        try:
            val = ast.literal_eval(val)
        except Exception:
            return []
    
    if isinstance(val, (int, float)):
        return [int(val)]
    
    if isinstance(val, list):
        return [int(float(x)) for x in val if x is not None and not (isinstance(x, float) and math.isnan(x))]
    
    return []

def clean_dataframe(df):
    df["rand_word_idx"] = df["rand_word_idx"].apply(ensure_int_list)
    df["rand_sent_idx"] = df["rand_sent_idx"].apply(ensure_int_list)
    return df

