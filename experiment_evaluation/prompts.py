output_prompt_classification = ' Use JSON format with the keys as "id", "prediction". Return only the JSON object with the root named "samples". Do not change the value of the "id".'

output_prompt_generation = ' Use JSON format with the keys as "id", "pred_summary". Return only the JSON object with the root named "samples". Do not change the value of the "id".'

bangla_hate_speech_base = '''
You are an expert in natural language processing and hate speech detection. Your task is to analyze Bengali text and determine whether it contains hate speech.

Follow these steps:
1. Read the input text carefully.
2. Identify if the text contains hate speech.
3. Classify the text into one of the following categories:
   - "Hate": The text contains hateful, offensive, or harmful language.
   - "Non-Hate": The text does not contain hateful or offensive language.

You will receive an array of samples, each containing "id" and "text".
'''

codemixed_hate_speech_base = '''
You are an expert in natural language processing and hate speech detection. Your task is to analyze codemixed Bengali text and determine whether it contains hate speech.

Follow these steps:
1. Read the input text carefully.
2. Identify if the text contains hate speech.
3. Classify the text into one of the following categories:
   - "Hate": The text contains hateful, offensive, or harmful language.
   - "Non-Hate": The text does not contain hateful or offensive language.

You will receive an array of samples, each containing "id" and codemixed "text".
'''

bangla_fake_news_base = """
You are an expert in natural language processing and fake news detection. Your task is to analyze Bengali text and determine whether it is likely to be fake news. Fake news refers to false or misleading information or satire content presented as news, often with the intent to deceive or manipulate.
For each of the sample you will receive the headline followed by the content.
Follow these steps:
1. Read the input text carefully.
2. Analyze the content for signs of fake news, such as:
   - Sensational or exaggerated claims.
   - Lack of credible sources or evidence.
   - Contradictions with known facts or reliable information.
   - Emotional or manipulative language.
3. Classify the text into one of the following categories:
   - "Fake News": The text is likely to be false or misleading.
   - "Not Fake News": The text appears to be credible and factual.
4. Provide a brief explanation for your classification.

The following types of news are considered as fake news:

• Misleading/False Context: Any news with unreliable information or contains facts that can mislead audiences.
• Clickbait: News that uses sensitive headlines to grab attention and drive click-throughs to the publisher's website.
• Satire/Parody: News stories that are intended for entertainment and parody

You will receive an array of objects, each containing an 'id' and 'text'.
"""

codemixed_fake_news_base = """
You are an expert in natural language processing and fake news detection. Your task is to analyze codemixed Bengali text and determine whether it is likely to be fake news. Fake news refers to false or misleading information or satire content presented as news, often with the intent to deceive or manipulate.
For each of the sample you will receive the headline followed by the content.
Follow these steps:
1. Read the input text carefully.
2. Analyze the content for signs of fake news, such as:
   - Sensational or exaggerated claims.
   - Lack of credible sources or evidence.
   - Contradictions with known facts or reliable information.
   - Emotional or manipulative language.
3. Classify the text into one of the following categories:
   - "Fake News": The text is likely to be false or misleading.
   - "Not Fake News": The text appears to be credible and factual.
4. Provide a brief explanation for your classification.

The following types of news are considered as fake news:

• Misleading/False Context: Any news with unreliable information or contains facts that can mislead audiences.
• Clickbait: News that uses sensitive headlines to grab attention and drive click-throughs to the publisher's website.
• Satire/Parody: News stories that are intended for entertainment and parody

You will receive an array of objects, each containing an 'id' and 'text'.
"""

bangla_summarization_base = '''
You are an expert in natural language processing and text summarization. Your task is to summarize Bengali text into a concise and meaningful version while preserving the key points and overall meaning.

Follow these steps:
1. Read the input Bengali text carefully.
2. Identify the main ideas, key points, and essential information.
3. Write a summary that is shorter than the original text but retains the core meaning.

Ensure:
- Complete JSON structure
- No truncation
- Valid syntax

You will receive an array of samples, each containing "id" and "text".
'''

codemixed_summarization_base = '''
You are an expert in natural language processing and text summarization. Your task is to summarize codemixed Bengali text into a concise and meaningful version while preserving the key points and overall meaning.

Follow these steps:
1. Read the input Bengali text carefully.
2. Identify the main ideas, key points, and essential information.
3. Write a summary that is shorter than the original text but retains the core meaning.

Ensure:
- Complete JSON structure
- No truncation
- Valid syntax

You will receive an array of samples, each containing "id" and "text".
'''

BANGLA_HATE_SPEECH_PROMPT = bangla_hate_speech_base + output_prompt_classification
CODEMIXED_HATE_SPEECH_PROMPT = codemixed_hate_speech_base + output_prompt_classification
BANGLA_FAKE_NEWS_PROMPT = bangla_fake_news_base + output_prompt_classification
CODEMIXED_FAKE_NEWS_PROMPT = codemixed_fake_news_base + output_prompt_classification
BANGLA_SUMMARIZATION_PROMPT = bangla_summarization_base + output_prompt_generation
CODEMIXED_SUMMARIZATION_PROMPT = codemixed_summarization_base + output_prompt_generation

