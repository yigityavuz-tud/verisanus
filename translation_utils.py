import pandas as pd
import deepl
import requests
import json
import os
from pathlib import Path
import yaml

def load_config():
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

def get_deepl_token():
    config = load_config()['google_maps']
    token_file = config['api_settings']['deepl_token_file']
    try:
        with open(token_file, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        raise ValueError(f"DeepL token file {token_file} not found. Please check the config file.")

def get_deepseek_token():
    config = load_config()['google_maps']
    token_file = config['api_settings']['deepseek_token_file']
    try:
        with open(token_file, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        raise ValueError(f"DeepSeek token file {token_file} not found. Please check the config file.")

def count_characters(text):
    """Count the number of characters in a text, handling NaN values."""
    if pd.isna(text):
        return 0
    return len(str(text))

def translate_with_deepseek(text, api_key):
    """
    Translate text using DeepSeek API.
    """
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    # Craft a prompt that will ensure we get just the translation
    prompt = f"Translate the following text to English. Only provide the translation, nothing else:\n\n{text}"
    
    data = {
        "messages": [{"role": "user", "content": prompt}],
        "model": "deepseek-chat",
        "temperature": 0.1  # Low temperature for more consistent translations
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error using DeepSeek API: {str(e)}")
        return None

def translate_column(
    input_data,
    source_column,
    target_column=None,
    target_language="EN-GB",
    translation_service=None,
    suffix="_en",
    output_path=None
):
    """
    Translate a column in a DataFrame or Excel file using either DeepL or DeepSeek.
    
    Parameters:
    -----------
    input_data : str or pandas.DataFrame
        Either a path to an Excel file or a pandas DataFrame
    source_column : str
        Name of the column to translate
    target_column : str, optional
        Name of the column to store translations. If None, will use source_column + suffix
    target_language : str, optional
        Target language code (default: "EN-GB")
    translation_service : str, optional
        Either "deepl" or "deepseek". If None, will use the service from config
    suffix : str, optional
        Suffix to add to source_column if target_column is None (default: "_en")
    output_path : str, optional
        Path to save the translated DataFrame. If None, the DataFrame will not be saved.
    
    Returns:
    --------
    tuple
        (DataFrame with translations, number of characters translated)
    """
    # Load the data
    if isinstance(input_data, str):
        df = pd.read_excel(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise ValueError("input_data must be either a file path (str) or a pandas DataFrame")
    
    # Determine target column name
    if target_column is None:
        target_column = f"{source_column}{suffix}"
    
    # Initialize target column if it doesn't exist
    if target_column not in df.columns:
        df[target_column] = df[source_column]
    
    # Get translation service from config if not specified
    if translation_service is None:
        config = load_config()['google_maps']
        translation_service = config['api_settings'].get('translation_service', 'deepl')
    
    # Initialize translation service
    translator = None
    deepseek_token = None
    
    if translation_service == 'deepl':
        translator = deepl.Translator(get_deepl_token())
    else:  # deepseek
        deepseek_token = get_deepseek_token()
    
    # Get rows that need translation
    to_translate = df[
        (df[source_column].notna()) & 
        (df[source_column] != '')
    ].index
    
    characters_translated = 0
    
    if not to_translate.empty:
        print(f"Translating {len(to_translate)} rows using {translation_service.upper()}...")
        for idx in to_translate:
            try:
                if translation_service == 'deepseek':
                    translated_text = translate_with_deepseek(
                        df.loc[idx, source_column], 
                        deepseek_token
                    )
                    if translated_text:
                        df.loc[idx, target_column] = translated_text
                        characters_translated += count_characters(df.loc[idx, source_column])
                else:  # Use DeepL
                    translated = translator.translate_text(
                        df.loc[idx, source_column], 
                        target_lang=target_language
                    )
                    df.loc[idx, target_column] = translated.text
                    characters_translated += count_characters(df.loc[idx, source_column])
            except Exception as e:
                print(f"Error translating text at index {idx}: {str(e)}")
                continue
    
    # Save the file with translations if output_path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_excel(output_path, index=False)
        print(f"Translations saved to {output_path}")
    
    print(f"Total characters translated: {characters_translated}")
    
    if translation_service == 'deepl':
        log_deepl_usage(characters_translated)
    
    return df, characters_translated

def log_deepl_usage(characters_translated):
    """Log the total characters translated to a file."""
    log_file = "tokens/deepl_usage_log.txt"
    
    # Get current date
    from datetime import datetime
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Read existing log if it exists
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            lines = f.readlines()
            if lines:
                last_line = lines[-1].strip()
                try:
                    total_chars = int(last_line.split(': ')[1])
                except (IndexError, ValueError):
                    total_chars = 0
            else:
                total_chars = 0
    else:
        total_chars = 0
    
    # Add new characters to total
    total_chars += characters_translated
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Append new log entry
    with open(log_file, 'a') as f:
        f.write(f"{current_date}: {total_chars}\n")
    
    return total_chars 