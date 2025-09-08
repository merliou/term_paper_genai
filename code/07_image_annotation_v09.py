# 02_image_annotation.py

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
# --- KONFIGURATION ---
# ==============================================================================
# Eingabedatei, die von '01_text_classification.py' erstellt wurde
INPUT_CSV_FILE = 'annotations/subset_2_text_classified_qwen3:4b_3gb.csv'
PROCESSING_CSV_FILE = 'annotations/subset_2_qwen3:4b_x_llava:7b' 

IMAGE_BATCH_SIZE = 1
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
IMAGE_MODEL = "llava:7b"
API_TIMEOUT = 660
IMAGE_PROMPT_PATH = "term_paper_genai/prompts/02_image_annotation_prompt_v04.txt"
IMAGE_DPI = 96
IMAGE_GRAYSCALE = True
IMAGE_QUALITY = 80

# ==============================================================================
# --- HILFSFUNKTIONEN ---
# ==============================================================================
def load_prompt(file_path):
    """Lädt einen Prompt aus einer Textdatei."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f: return f.read()
    except FileNotFoundError:
        print(f"FATALER FEHLER: Prompt-Datei nicht gefunden: {file_path}"); return None

def call_ollama_api_vision(prompt, model, image_bytes):
    """Ruft die Ollama API für Bild-Annotation auf."""
    payload = {"model": model, "prompt": prompt, "format": "json", "stream": False}
    if image_bytes:
        payload["images"] = [base64.b64encode(image_bytes).decode('utf-8')]
    try:
        response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=API_TIMEOUT)
        response.raise_for_status()
        response_text = response.json().get('response', '{}')
        return json.loads(response_text)
    except (requests.RequestException, json.JSONDecodeError) as e:
        print(f"  -> API-Fehler oder JSON-Decode-Fehler: {e}"); return {"error": str(e)}

# ==============================================================================
# --- WORKFLOW-SCHRITT ---
# ==============================================================================
def step3_annotate_images_in_batches(df):
    """Annotiert die als relevant markierten Seiten im Batch-Modus."""
    print("\n--- SCHRITT 3: Annotiere markierte Seiten (Batch-Modus) ---")
    image_prompt = load_prompt(IMAGE_PROMPT_PATH)
    if not image_prompt: return df
    
    annotation_cols = ['alc', 'product', 'warning', 'discount']
    for col in annotation_cols:
        if col not in df.columns: df[col] = pd.NA
        
    # Identifiziere Seiten, die annotiert werden müssen
    needs_annotation_mask = (df['alc_keyword_flag'] == 1)
    already_annotated_mask = df['alc'].notna()
    to_process_indices = df[needs_annotation_mask & ~already_annotated_mask].index
    
    if to_process_indices.empty:
        print("Alle als relevant markierten Seiten scheinen bereits annotiert zu sein. Nichts zu tun.")
        return df
        
    print(f"Insgesamt {len(to_process_indices)} Seiten müssen noch annotiert werden.")
    
    num_batches = (len(to_process_indices) + IMAGE_BATCH_SIZE - 1) // IMAGE_BATCH_SIZE
    
    for i in range(0, len(to_process_indices), IMAGE_BATCH_SIZE):
        batch_indices = to_process_indices[i : i + IMAGE_BATCH_SIZE]
        current_batch_num = (i // IMAGE_BATCH_SIZE) + 1
        
        print(f"\n--- Bearbeite Bild-Batch {current_batch_num} von {num_batches} (Seiten: {len(batch_indices)}) ---")
        
        for index in batch_indices:
            row = df.loc[index]
            print(f"  Annotiere Bild: {os.path.basename(row['page_pdf_path'])}")
            
            try:
                with fitz.open(row['page_pdf_path']) as doc:
                    pix = doc.load_page(0).get_pixmap(dpi=IMAGE_DPI, colorspace=fitz.csGRAY if IMAGE_GRAYSCALE else fitz.csRGB)
                    img = Image.frombytes("L" if IMAGE_GRAYSCALE else "RGB", [pix.width, pix.height], pix.samples)
                    buffer = BytesIO()
                    img.save(buffer, format="JPEG", quality=IMAGE_QUALITY)
                    image_bytes = buffer.getvalue()
            except Exception as e:
                print(f"    -> Fehler beim Rendern des Bildes: {e}"); continue
                
            annotation = call_ollama_api_vision(image_prompt, IMAGE_MODEL, image_bytes=image_bytes)
            
            if not annotation.get("error"):
                for col in annotation_cols:
                    df.loc[index, col] = annotation.get(col, pd.NA)
                print(f"    -> Annotation erfolgreich.")
            else:
                print(f"    -> Fehler bei der Annotation vom Vision-LLM erhalten.")
                
        print("\n-> Batch abgeschlossen. Speichere Zwischenergebnis...")
        df.to_csv(PROCESSING_CSV_FILE, index=False, encoding='utf-8-sig')
        print(f"   Erfolgreich in '{PROCESSING_CSV_FILE}' gespeichert.")
        
        # Zeige die Ergebnisse dieses Batches an
        display_cols = ['page_pdf_path'] + annotation_cols
        display_df = df.loc[batch_indices][display_cols].copy()
        display_df['page_pdf_path'] = display_df['page_pdf_path'].apply(os.path.basename)
        print("-> Ergebnisse dieses Batches:\n" + display_df.to_string())
        
        # *** WICHTIGE ÄNDERUNG: Die manuelle Bestätigung wurde entfernt ***
        # Das Skript läuft nun automatisch weiter zum nächsten Batch.
        
    print("\nAlle Bild-Batches wurden verarbeitet.")
    return df

# ==============================================================================
# --- HAUPTSKRIPT ---
# ==============================================================================
if __name__ == "__main__":
    print("Starte Workflow: Bild-Annotation.")

    # Prüfe, ob die Eingabedatei existiert
    if not os.path.exists(INPUT_CSV_FILE):
        print(f"FATALER FEHLER: Eingabedatei nicht gefunden: '{INPUT_CSV_FILE}'")
        print("Bitte führe zuerst das Skript '01_text_classification.py' aus.")
        exit()

    print(f"Lade klassifizierte Daten aus: '{INPUT_CSV_FILE}'")
    df = pd.read_csv(INPUT_CSV_FILE)

    # Führe den Annotationsschritt durch
    df_final = step3_annotate_images_in_batches(df)
    
    print(f"\n==========================================================")
    print(f"Workflow abgeschlossen! Der finale Stand wurde gespeichert in:")
    print(PROCESSING_CSV_FILE)
    print(f"==========================================================")