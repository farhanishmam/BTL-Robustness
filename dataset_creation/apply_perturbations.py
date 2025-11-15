import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from normalizer import normalize
from perturbation import perturb_random_words, perturb_random_sentences, perturb_salient_words

def load_transliteration_model(model_name="FabihaHaider/transliterated_nmt"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(torch_device)
    print(f"Using device: {torch_device}")
    return tokenizer, model, torch_device

def predict_output(input_sentence, tokenizer, model, torch_device):
    input_sentence = normalize(input_sentence)
    input_ids = tokenizer((input_sentence), return_tensors="pt").input_ids.to(torch_device)
    generated_tokens = model.generate(input_ids)
    decoded_tokens = tokenizer.batch_decode(generated_tokens)[0]
    decoded_tokens = normalize(decoded_tokens)
    decoded_tokens = decoded_tokens.replace("<pad>", "").replace("</s>", "").strip()
    return decoded_tokens

def apply_all_perturbations(input_file, output_file, text_column, batch_size=32, sample_size=None):
    df = pd.read_csv(input_file)
    
    if sample_size:
        df0 = df[df['label'] == 0].sample(sample_size // 2, random_state=42)
        df1 = df[df['label'] == 1].sample(sample_size // 2, random_state=42)
        df = pd.concat([df0, df1]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    tokenizer, model, torch_device = load_transliteration_model()
    
    predict_fn = lambda x: predict_output(x, tokenizer, model, torch_device)
    
    df_perturbed = (df
        .pipe(perturb_random_words, predict_fn, text_column, batch_size)
        .pipe(perturb_random_sentences, predict_fn, text_column, batch_size)
        .pipe(perturb_salient_words, predict_fn, text_column, batch_size)
    )
    
    df_perturbed.to_csv(output_file, index=False)
    print(f"Perturbed dataset saved to {output_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--text_column', type=str, required=True, help='Text column name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--sample_size', type=int, default=None, help='Sample size (optional)')
    args = parser.parse_args()
    
    apply_all_perturbations(args.input, args.output, args.text_column, args.batch_size, args.sample_size)

