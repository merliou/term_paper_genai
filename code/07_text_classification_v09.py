# 01_text_classification.py

import pandas as pd
import os
import time
import base64
import json
import requests
import fitz  # PyMuPDF

# ==============================================================================
# --- KONFIGURATION ---
# ==============================================================================
# Eingabedatei mit den Pfaden zu den PDFs
BASE_CSV_FILE = 'subsets/subset_2.csv'
# Ausgabedatei, die die Ergebnisse dieses Skripts enthält
OUTPUT_CSV_FILE = 'annotations/subset_2_text_classified_qwen3:4b_3gb.csv'
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
TEXT_MODEL = "qwen3:4b"
API_TIMEOUT = 240 # Timeout für Text-API-Calls
TEXT_PROMPT_PATH = "term_paper_genai/prompts/01_text_annotation_prompt_v03.txt" 

# ==============================================================================
# --- HILFSFUNKTIONEN ---
# ==============================================================================
def load_prompt(file_path):
    """Lädt einen Prompt aus einer Textdatei."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f: return f.read()
    except FileNotFoundError:
        print(f"FATALER FEHLER: Prompt-Datei nicht gefunden: {file_path}")
        return None

def call_ollama_api(prompt, model):
    """Ruft die Ollama API für Text-Klassifizierung auf."""
    payload = {"model": model, "prompt": prompt, "format": "json", "stream": False}
    try:
        response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=API_TIMEOUT)
        response.raise_for_status()
        response_text = response.json().get('response', '{}')
        return json.loads(response_text)
    except (requests.RequestException, json.JSONDecodeError) as e:
        print(f"  -> API-Fehler oder JSON-Decode-Fehler: {e}")
        return {"error": str(e)}

# ==============================================================================
# --- WORKFLOW-SCHRITTE ---
# ==============================================================================
def step1_extract_text(df):
    """Extrahiert den Text von jeder PDF-Seite in der DataFrame."""
    print("\n--- SCHRITT 1: Extrahiere Text aus allen PDF-Seiten ---")
    texts = []
    total = len(df)
    for index, row in df.iterrows():
        print(f"  Verarbeite Text von Seite {index + 1}/{total}...", end='\r')
        try:
            with fitz.open(row['page_pdf_path']) as doc:
                texts.append(doc.load_page(0).get_text("text"))
        except Exception as e:
            print(f"  Fehler bei {row['page_pdf_path']}: {e}")
            texts.append("")
    df['extracted_text'] = texts
    print(f"\nText-Extraktion für {total} Seiten abgeschlossen.")
    return df

def step2_classify_text(df):
    """Klassifiziert die extrahierten Texte mithilfe des Text-LLMs."""
    print("\n--- SCHRITT 2: Klassifiziere Texte auf Alkohol-Stichworte ---")
    text_prompt_template = load_prompt(TEXT_PROMPT_PATH)
    if not text_prompt_template: return df
    
    flags = []
    total = len(df)
    api_errors = 0

    for index, row in df.iterrows():
        print(f"  Klassifiziere Text {index + 1}/{total} (API-Fehler: {api_errors})...", end='\r')
        text = row['extracted_text']
        
        if pd.isna(text) or len(text.strip()) < 10:
            flags.append(0)
            continue
            
        truncated_text = text[:3000] # Text zur Sicherheit kürzen
        prompt = text_prompt_template.replace("{page_text}", truncated_text)
        result = call_ollama_api(prompt, TEXT_MODEL)
        
        if result.get("error"):
            api_errors += 1
            flags.append(0)
        else:
            # .get('flag', 0) stellt sicher, dass wir 0 erhalten, wenn 'flag' fehlt
            flags.append(result.get('flag', 0))
            
    df['alc_keyword_flag'] = flags
    print(f"\nText-Klassifizierung abgeschlossen.")
    if api_errors > 0:
        print(f"WARNUNG: Es gab {api_errors} API-Fehler (z.B. Timeouts). Diese Seiten wurden als 'nicht relevant' (0) markiert.")
    print(f"Insgesamt wurden {df['alc_keyword_flag'].sum()} von {total} Seiten als potenziell relevant markiert.")
    return df

# ==============================================================================
# --- HAUPTSKRIPT ---
# ==============================================================================
if __name__ == "__main__":
    print("Starte Workflow: Text-Extraktion und Klassifizierung.")
    
    # Lade die Basis-CSV-Datei
    try:
        df = pd.read_csv(BASE_CSV_FILE)
        print(f"Basis-Datei '{BASE_CSV_FILE}' erfolgreich geladen.")
    except FileNotFoundError:
        print(f"FATALER FEHLER: Basis-CSV-Datei nicht gefunden: {BASE_CSV_FILE}")
        exit()

    # Führe die Schritte aus
    df = step1_extract_text(df)
    df = step2_classify_text(df)

    # Erstelle das Ausgabe-Verzeichnis, falls es nicht existiert
    output_dir = os.path.dirname(OUTPUT_CSV_FILE)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Speichere die Ergebnisse
    df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8-sig')
    
    print(f"\n==========================================================")
    print(f"Workflow abgeschlossen! Die Ergebnisse wurden gespeichert in:")
    print(OUTPUT_CSV_FILE)
    print(f"Diese Datei kann nun mit dem Skript '02_image_annotation.py' verwendet werden.")
    print(f"==========================================================")