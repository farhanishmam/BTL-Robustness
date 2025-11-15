import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from normalizer import normalize
from bnlp import BasicTokenizer
from dataset_utils import bangla_punctuations, all_punctuation

class BanglaBertAttentionAnalyzer:
    def __init__(self, model_name="sagorsarker/bangla-bert-base", prob=0.2):
        self.prob = prob
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("csebuetnlp/banglabert")
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def get_attention_scores(self, text):
        text = normalize(text)
        
        basic_tokenizer = BasicTokenizer()
        pre_tokens = basic_tokenizer.tokenize(text)
        pre_tokenized_text = " ".join(pre_tokens)
        
        inputs = self.tokenizer(
            pre_tokenized_text,
            return_tensors="pt",
            add_special_tokens=True,
            max_length=512,
            truncation=True
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        attention = outputs.attentions
        attention_scores = torch.mean(torch.stack(attention), dim=(0, 2))
        attention_scores = attention_scores.cpu().numpy()[0]
        token_attention = attention_scores.mean(axis=0)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        tokens = [normalize(token) for token in tokens]
        return tokens, token_attention

    def get_salient_words(self, text):
        top_k = max(1, round(len(text.split()) * self.prob))
        tokens, attention_scores = self.get_attention_scores(text)
        
        word_scores = {}
        current_word = ""
        current_score = 0
        
        for token, score in zip(tokens, attention_scores):
            if token.startswith("##"):
                current_word += token[2:]
                current_score += score
            else:
                if current_word:
                    word_scores[current_word] = current_score
                current_word = token
                current_score = score
        
        if current_word:
            word_scores[current_word] = current_score
        
        remove_special_token = ['[CLS]', '[SEP]', '[PAD]'] + bangla_punctuations
        for special_token in remove_special_token:
            word_scores.pop(special_token, None)
        
        salient_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return salient_words

    def visualize_attention(self, text):
        tokens, attention_scores = self.get_attention_scores(text)
        normalized_scores = (attention_scores - attention_scores.min()) / (attention_scores.max() - attention_scores.min())
        
        html_text = ""
        for token, score in zip(tokens, normalized_scores):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                continue
            if token.startswith("##"):
                token = token[2:]
            intensity = int(score * 255)
            html_text += f'<span style="background-color: rgba(255, 165, 0, {score:.2f})">{token}</span>'
        
        return html_text

