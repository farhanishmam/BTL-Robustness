import os
import pandas as pd
import json
import time
from evaluation_metrics import calculate_scores_generation_task

def process_text(text):
    processed = text.replace('\n', ' ').strip()
    while '  ' in processed:
        processed = processed.replace('  ', ' ')
    return processed

def create_chunks(df, column, chunk_size=10):
    chunk_list = []
    total_rows = len(df)
    num_chunks = (total_rows + chunk_size - 1) // chunk_size
    
    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, total_rows)
        
        chunk_df = df.iloc[start_idx:end_idx]
        
        text_entries = []
        for _, row in chunk_df.iterrows():
            clean_text = process_text(str(row[column]))
            text_entries.append(f"{row.name}: {clean_text}")
        
        chunk_dict = {
            'chunk_name': f"{chunk_df.index[0]}-{chunk_df.index[-1]}",
            'user_prompt': '\n'.join(text_entries)
        }
        
        chunk_list.append(chunk_dict)
    
    return chunk_list

def run_claude_generation(df, column, system_prompt, client, chunk_size=5):
    total_cost = 0
    
    def get_sentiment_analysis(data, system, prefill=""):
        max_retries = 3
        retry_delay = 60
        
        for attempt in range(max_retries):
            try:
                message = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=2000,
                    temperature=0.0,
                    system=system,
                    messages=[
                        {"role": "user", "content": data},
                        {"role": "assistant", "content": prefill}
                    ]
                )
                
                input_tokens = message.usage.input_tokens
                output_tokens = message.usage.output_tokens
                
                nonlocal total_cost
                input_cost = input_tokens * 0.000003
                output_cost = output_tokens * 0.000015
                total_cost += input_cost + output_cost
                
                response_text = message.content[0].text
                print(response_text)
                return response_text
                
            except Exception as e:
                if "RateLimitError" in str(type(e)) and attempt < max_retries - 1:
                    print(f"Rate limit hit. Waiting {retry_delay} seconds before retry...")
                    time.sleep(retry_delay)
                else:
                    raise
    
    chunk_list = create_chunks(df, column, chunk_size)
    final_df = pd.DataFrame()
    
    for chunk in chunk_list:
        try:
            data = chunk['user_prompt']
            response = get_sentiment_analysis(data, system_prompt)
            parsed_data = json.loads(response)
            results = parsed_data['samples']
            
            output_df = pd.DataFrame(results)
            final_df = pd.concat([final_df, output_df], ignore_index=True)
            
            time.sleep(2)
            
        except Exception as e:
            print(f"An error occurred for chunk {chunk['chunk_name']}: {e}")
            continue
    
    print(f"Total cost incurred: ${total_cost:.4f}")
    
    final_df.drop_duplicates(subset='id', keep='first', inplace=True)
    save_df = pd.merge(df, final_df, on='id', how='inner')
    
    return save_df

def run_gpt_generation(df, column, system_prompt, client, chunk_size=10):
    def summarize(system, data):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": data}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "summarization_response",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "samples": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "id": {"type": "integer"},
                                        "summary": {"type": "string"}
                                    },
                                    "required": ["id", "summary"],
                                    "additionalProperties": False
                                }
                            }
                        },
                        "required": ["samples"],
                        "additionalProperties": False
                    }
                }
            }
        )
        return response.choices[0].message.content
    
    chunk_list = create_chunks(df, column, chunk_size)
    final_df = pd.DataFrame()
    
    for chunk in chunk_list:
        try:
            response = summarize(system_prompt, chunk['user_prompt'])
            response_object = json.loads(response)
            output_df = pd.DataFrame(response_object['samples'])
            final_df = pd.concat([final_df, output_df], ignore_index=True)
        except Exception as e:
            print(f"An error occurred: {e} for chunk {chunk['chunk_name']}")
    
    result_df = pd.merge(df, final_df, on='id', how='inner')
    result_df.dropna(inplace=True)
    
    return result_df

