import pandas as pd
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction, closest_ref_length, brevity_penalty
from nltk.util import ngrams
from sklearn.metrics import classification_report, accuracy_score
from normalizer import normalize

def download_nltk_resources():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
    except:
        pass

download_nltk_resources()

def calculate_bleu(reference_sentence, candidate_sentence):
    reference = [word_tokenize(reference_sentence)]
    candidate = word_tokenize(candidate_sentence)
    
    smoothing_function = SmoothingFunction().method1
    
    bl = sentence_bleu(reference, candidate, weights=(1, 0.5, 0.33, 0.25), smoothing_function=smoothing_function)
    
    hyp_len = len(candidate)
    ref_len = len(reference[0])
    closest_ref_len = closest_ref_length(reference, hyp_len)
    bp = brevity_penalty(closest_ref_len, hyp_len)
    
    ratio = hyp_len / ref_len if ref_len > 0 else 0
    
    return bl, bp, ratio

def run_bleu_script(df, true_col, pred_col):
    total_bleu = 0
    total_bp = 0
    total_ratio = 0
    for reference_sentence, candidate_sentence in zip(df[true_col], df[pred_col]):
        bleu, bp, ratio = calculate_bleu(str(reference_sentence), str(candidate_sentence))
        total_bleu += bleu
        total_bp += bp
        total_ratio += ratio
    
    bleu = total_bleu / df.shape[0]
    bp = total_bp / df.shape[0]
    ratio = total_ratio / df.shape[0]
    print(f"bleu: {bleu}, bp: {bp}, ratio: {ratio}")
    return {'bleu': bleu, 'bp': bp, 'ratio': ratio}

def calculate_rouge_scores(reference_tokens, system_tokens):
    def lcs(X, Y):
        m, n = len(X), len(Y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if X[i - 1] == Y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]
    
    reference_unigrams = set(reference_tokens)
    system_unigrams = set(system_tokens)
    overlap_rouge1 = len(reference_unigrams.intersection(system_unigrams))
    precision_rouge1 = overlap_rouge1 / len(system_unigrams) if len(system_unigrams) > 0 else 0
    recall_rouge1 = overlap_rouge1 / len(reference_unigrams) if len(reference_unigrams) > 0 else 0
    r1_t = 1 if precision_rouge1 + recall_rouge1 == 0 else 0
    f1_rouge1 = 2 * (precision_rouge1 * recall_rouge1) / (precision_rouge1 + recall_rouge1 + r1_t)
    
    reference_bigrams = set(ngrams(reference_tokens, 2))
    system_bigrams = set(ngrams(system_tokens, 2))
    
    overlap_rouge2 = len(reference_bigrams.intersection(system_bigrams))
    precision_rouge2 = overlap_rouge2 / len(system_bigrams) if len(system_bigrams) > 0 else 0
    recall_rouge2 = overlap_rouge2 / len(reference_bigrams) if len(reference_bigrams) > 0 else 0
    r2_t = 1 if precision_rouge2 + recall_rouge2 == 0 else 1
    f1_rouge2 = 2 * (precision_rouge2 * recall_rouge2) / (precision_rouge2 + recall_rouge2 + r2_t)
    
    lcs_length = lcs(reference_tokens, system_tokens)
    precision_rougeL = lcs_length / len(system_tokens) if len(system_tokens) > 0 else 0
    recall_rougeL = lcs_length / len(reference_tokens) if len(reference_tokens) > 0 else 0
    rL_t = 1 if precision_rougeL + recall_rougeL == 0 else 0
    f1_rougeL = 2 * (precision_rougeL * recall_rougeL) / (precision_rougeL + recall_rougeL + rL_t)
    
    return {
        'ROUGE-1 Precision': precision_rouge1,
        'ROUGE-1 Recall': recall_rouge1,
        'ROUGE-1 F1': f1_rouge1,
        'ROUGE-2 Precision': precision_rouge2,
        'ROUGE-2 Recall': recall_rouge2,
        'ROUGE-2 F1': f1_rouge2,
        'ROUGE-L Precision': precision_rougeL,
        'ROUGE-L Recall': recall_rougeL,
        'ROUGE-L F1': f1_rougeL,
    }

def calculate_average_rouge_scores(reference_texts, system_texts):
    total_scores = {
        'ROUGE-1 Precision': 0,
        'ROUGE-1 Recall': 0,
        'ROUGE-1 F1': 0,
        'ROUGE-2 Precision': 0,
        'ROUGE-2 Recall': 0,
        'ROUGE-2 F1': 0,
        'ROUGE-L Precision': 0,
        'ROUGE-L Recall': 0,
        'ROUGE-L F1': 0,
    }
    
    num_pairs = len(reference_texts)
    
    for i in range(num_pairs):
        reference_text = reference_texts[i]
        system_text = system_texts[i]
        
        reference_tokens = nltk.word_tokenize(reference_text)
        system_tokens = nltk.word_tokenize(system_text)
        
        scores = calculate_rouge_scores(reference_tokens, system_tokens)
        
        for key, value in scores.items():
            total_scores[key] += value
    
    average_scores = {key: value / num_pairs for key, value in total_scores.items()}
    
    return average_scores

def calculate_rouge(df, true_col, pred_col):
    reference_texts = [normalize(str(sentence)) for sentence in df[true_col].tolist()]
    system_texts = [normalize(str(sentence)) for sentence in df[pred_col].tolist()]
    
    average_scores = calculate_average_rouge_scores(reference_texts, system_texts)
    print("Average ROUGE Scores:")
    print("-" * 30)
    for key, value in average_scores.items():
        print(key + ": {:.4f}".format(value))

def calculate_scores_generation_task(result_df, true_col='summary_x', pred_col='summary_y'):
    calculate_rouge(result_df, true_col, pred_col)
    run_bleu_script(result_df, true_col, pred_col)

def calculate_classification_metrics(y_true, y_pred):
    report = classification_report(y_true, y_pred, digits=4)
    print(report)
    return report

