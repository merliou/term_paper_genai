import pandas as pd
import os
import time
import base64
import json
import requests
import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO

# ==============================================================================
# --- KONFIGURATION (IHRE ÄNDERUNGEN WURDEN ÜBERNOMMEN) ---
# ==============================================================================
PROCESSING_CSV_FILE = 'annotations/subset_1_annotated_twosteps_v04.csv'
BASE_CSV_FILE = 'subsets/subset_1.csv'

# Konfiguration für Schritt 2 & 3: Batch-Verarbeitung
TEXT_BATCH_SIZE = 10 
IMAGE_BATCH_SIZE = 3

# Ollama Konfiguration
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"

# *** WICHTIGSTE ÄNDERUNG: DAS RICHTIGE MODELL FÜR DIE AUFGABE WÄHLEN ***
# EMPFEHLUNG: Ein kleines, schnelles Text-Modell für die Text-Klassifizierung
TEXT_MODEL = "mistral:7b-instruct-v0.3-q4_K_M" 
# Ihr multimodales Modell für die Bild-Analyse
IMAGE_MODEL = "qwen2.5vl:3b"

API_TIMEOUT = 1320 # Ihr Timeout-Wert

# Pfade zu den Prompt-Dateien
TEXT_PROMPT_PATH = "term_paper_genai/prompts/01_text_classification_batch_prompt.txt"
IMAGE_PROMPT_PATH = "term_paper_genai/prompts/02_image_annotation_prompt.txt"

# Konfiguration für die Bildverarbeitung
IMAGE_DPI = 96
IMAGE_GRAYSCALE = True
IMAGE_QUALITY = 80

# ==============================================================================
# --- HILFSFUNKTIONEN (unverändert) ---
# ==============================================================================
def load_prompt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f: return f.read()
    except FileNotFoundError:
        print(f"FATALER FEHLER: Prompt-Datei nicht gefunden: {file_path}"); return None

def call_ollama_api(prompt, model, image_bytes=None):
    payload = {"model": model, "prompt": prompt, "format": "json", "stream": False}
    if image_bytes:
        payload["images"] = [base64.b64encode(image_bytes).decode('utf-8')]
    try:
        # Verwende ein kürzeres Timeout für den schnellen Text-Call, um Fehler schneller zu erkennen
        timeout = 60 if image_bytes is None else API_TIMEOUT
        response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=timeout)
        response.raise_for_status()
        response_text = response.json().get('response', '{}')
        return json.loads(response_text)
    except (requests.RequestException, json.JSONDecodeError) as e:
        print(f"  -> API-Fehler oder JSON-Decode-Fehler: {e}"); return {"error": str(e)}

# ==============================================================================
# --- WORKFLOW-SCHRITTE (SCHRITT 2 leicht angepasst) ---
# ==============================================================================
def step1_extract_text(df):
    print("\n--- SCHRITT 1: Extrahiere Text aus allen PDF-Seiten ---")
    texts = []
    for index, row in df.iterrows():
        print(f"  Verarbeite Text von Seite {index + 1}/{len(df)}...", end='\r')
        try:
            with fitz.open(row['page_pdf_path']) as doc:
                texts.append(doc.load_page(0).get_text("text"))
        except Exception as e:
            print(f"  Fehler bei {row['page_pdf_path']}: {e}"); texts.append("")
    df['extracted_text'] = texts
    print(f"\nText-Extraktion für {len(df)} Seiten abgeschlossen.")
    return df

def step2_classify_text_in_batches(df):
    print("\n--- SCHRITT 2: Klassifiziere Texte auf Alkohol-Stichworte (Batch-Modus) ---")
    text_prompt_template = load_prompt(TEXT_PROMPT_PATH)
    if not text_prompt_template: return df

    df['alc_keyword_flag'] = 0
    to_process_df = df[df['extracted_text'].str.len() > 10].copy()
    to_process_indices = to_process_df.index
    
    total_to_process = len(to_process_indices)
    num_batches = (total_to_process + TEXT_BATCH_SIZE - 1) // TEXT_BATCH_SIZE
    print(f"Verarbeite {total_to_process} Seiten mit Text in {num_batches} Batches...")

    for i in range(0, total_to_process, TEXT_BATCH_SIZE):
        batch_indices = to_process_indices[i : i + TEXT_BATCH_SIZE]
        current_batch_num = (i // TEXT_BATCH_SIZE) + 1
        print(f"  Verarbeite Batch {current_batch_num}/{num_batches}...")

        batch_data = [{"ID": index, "TEXT": df.loc[index, 'extracted_text']} for index in batch_indices]
        prompt = text_prompt_template.replace("{batch_of_texts}", json.dumps(batch_data, indent=2))
        
        result = call_ollama_api(prompt, TEXT_MODEL)
        
        if "results" in result and isinstance(result["results"], list):
            for item in result["results"]:
                if "ID" in item and "flag" in item:
                    df.loc[item["ID"], 'alc_keyword_flag'] = item["flag"]
        else:
            print(f"    -> Warnung: Unerwartetes Format vom LLM für Batch {current_batch_num}. Details: {result}")

    print(f"\nText-Klassifizierung abgeschlossen. {df['alc_keyword_flag'].sum()} Seiten markiert.")
    return df

def step3_annotate_images_in_batches(df):
    """Annotiert die markierten Seiten in kontrollierbaren Batches."""
    print("\n--- SCHRITT 3: Annotiere markierte Seiten (Batch-Modus) ---")
    image_prompt = load_prompt(IMAGE_PROMPT_PATH)
    if not image_prompt: return df

    # Ihre angepasste Spaltenliste wurde übernommen
    annotation_cols = ['alc', 'product', 'reduc', 'prod_pp', 'prod_pp_alc']
    for col in annotation_cols:
        if col not in df.columns: df[col] = pd.NA

    needs_annotation_mask = (df['alc_keyword_flag'] == 1)
    already_annotated_mask = df['alc'].notna()
    to_process_indices = df[needs_annotation_mask & ~already_annotated_mask].index

    if to_process_indices.empty:
        print("Alle als relevant markierten Seiten wurden bereits annotiert. Nichts zu tun.")
        return df

    print(f"Insgesamt {len(to_process_indices)} Seiten müssen noch annotiert werden.")
    
    for i in range(0, len(to_process_indices), IMAGE_BATCH_SIZE):
        batch_indices = to_process_indices[i : i + IMAGE_BATCH_SIZE]
        num_batches = (len(to_process_indices) + IMAGE_BATCH_SIZE - 1) // IMAGE_BATCH_SIZE
        current_batch_num = (i // IMAGE_BATCH_SIZE) + 1
        
        print(f"\n--- Bearbeite Bild-Batch {current_batch_num} von {num_batches} (Seiten: {len(batch_indices)}) ---")

        for index in batch_indices:
            row = df.loc[index]
            print(f"  Annotiere Bild: {os.path.basename(row['page_pdf_path'])}")
            try:
                with fitz.open(row['page_pdf_path']) as doc:
                    pix = doc.load_page(0).get_pixmap(dpi=IMAGE_DPI, colorspace=fitz.csGRAY if IMAGE_GRAYSCALE else fitz.csRGB)
                    img = Image.frombytes("L" if IMAGE_GRAYSCALE else "RGB", [pix.width, pix.height], pix.samples)
                    buffer = BytesIO(); img.save(buffer, format="JPEG", quality=IMAGE_QUALITY)
                    image_bytes = buffer.getvalue()
            except Exception as e:
                print(f"    -> Fehler beim Rendern des Bildes: {e}"); continue

            annotation = call_ollama_api(image_prompt, IMAGE_MODEL, image_bytes=image_bytes)

            if not annotation.get("error"):
                for col in annotation_cols:
                    df.loc[index, col] = annotation.get(col, pd.NA)
                print(f"    -> Annotation erfolgreich.")
            else:
                print(f"    -> Fehler bei der Annotation vom LLM erhalten.")
        
        print("\n-> Batch abgeschlossen. Speichere Zwischenergebnis...")
        df.to_csv(PROCESSING_CSV_FILE, index=False, encoding='utf-8-sig')
        print(f"   Erfolgreich in '{PROCESSING_CSV_FILE}' gespeichert.")
        
        display_cols = ['page_pdf_path'] + annotation_cols
        display_df = df.loc[batch_indices][display_cols].copy()
        display_df['page_pdf_path'] = display_df['page_pdf_path'].apply(os.path.basename)
        print("-> Ergebnisse dieses Batches:\n" + display_df.to_string())

        if current_batch_num < num_batches:
            user_input = input("\nDrücke Enter, um den nächsten Batch zu starten, oder 'q' zum Beenden: ")
            if user_input.lower() == 'q':
                print("Benutzer hat den Prozess beendet."); return df
    
    print("\nAlle Bild-Batches wurden verarbeitet.")
    return df

# ==============================================================================
# --- HAUPTSKRIPT (unverändert) ---
# ==============================================================================
if __name__ == "__main__":
    if os.path.exists(PROCESSING_CSV_FILE):
        print(f"Lade bestehende Arbeitsdatei: '{PROCESSING_CSV_FILE}'")
        df = pd.read_csv(PROCESSING_CSV_FILE)
    else:
        print(f"Arbeitsdatei nicht gefunden. Lade Basis-Datei: '{BASE_CSV_FILE}'")
        try:
            df = pd.read_csv(BASE_CSV_FILE)
        except FileNotFoundError:
            print(f"FATALER FEHLER: Basis-CSV-Datei nicht gefunden: {BASE_CSV_FILE}"); exit()

    print("Starte Annotations-Workflow...")

    if 'extracted_text' not in df.columns:
        df = step1_extract_text(df)
        df.to_csv(PROCESSING_CSV_FILE, index=False, encoding='utf-8-sig')
    else:
        print("\n--- SCHRITT 1: Text-Extraktion bereits abgeschlossen. Überspringe. ---")

    if 'alc_keyword_flag' not in df.columns:
        df = step2_classify_text_in_batches(df)
        df.to_csv(PROCESSING_CSV_FILE, index=False, encoding='utf--sig')
    else:
        print("\n--- SCHRITT 2: Text-Klassifizierung bereits abgeschlossen. Überspringe. ---")

    df_final = step3_annotate_images_in_batches(df)

    print(f"\n==========================================================")
    print(f"Workflow abgeschlossen! Der finale Stand wurde gespeichert in:")
    print(PROCESSING_CSV_FILE)
    print(f"==========================================================")