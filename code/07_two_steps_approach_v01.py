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

# Schalter für einen schnellen Testlauf mit nur den ersten 10 Seiten
TEST_RUN = False
TEST_RUN_SIZE = 10

# Eingabe- und Ausgabedateien
# Das Skript liest diese Datei und erstellt am Ende eine finale Version davon.
BASE_CSV_FILE = 'subsets/subset_2.csv'
FINAL_OUTPUT_CSV = 'annotations/subset_1_annotated_twosteps.csv' # Das Endergebnis wird hier gespeichert

# Ollama Konfiguration
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
# Modell für die schnelle Text-Klassifizierung (multilingual, klein)
TEXT_MODEL = "qwen2.5vl:3b" # z.B. Mistral oder ein anderes gutes Text-Modell
# Modell für die detaillierte Bild-Analyse
IMAGE_MODEL = "qwen2.5vl:3b" # Wie von Ihnen gewünscht (oder qwen2:0.5b für noch kleiner)

# Pfade zu den Prompt-Dateien
TEXT_PROMPT_PATH = "term_paper_genai/prompts/01_text_annotation_prompt.txt"
IMAGE_PROMPT_PATH = "term_paper_genai/prompts/02_image_annotation_prompt.txt"

# Konfiguration für die Bildverarbeitung
IMAGE_DPI = 96  # Niedrigere DPI für schnellere Verarbeitung und kleinere Bilder
IMAGE_GRAYSCALE = True # Graustufen sparen viel Platz
IMAGE_QUALITY = 80 # JPG-Qualität (80 ist ein guter Kompromiss)

# ==============================================================================
# --- HILFSFUNKTIONEN ---
# ==============================================================================

def load_prompt(file_path):
    """Lädt einen Prompt aus einer Textdatei."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"FATALER FEHLER: Prompt-Datei nicht gefunden: {file_path}")
        return None

def call_ollama_api(prompt, model, image_bytes=None):
    """Zentrale Funktion für alle Ollama API-Aufrufe."""
    payload = {"model": model, "prompt": prompt, "format": "json", "stream": False}
    if image_bytes:
        payload["images"] = [base64.b64encode(image_bytes).decode('utf-8')]

    try:
        response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=600)
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
    """Extrahiert Text aus jeder PDF-Seite und fügt ihn als neue Spalte hinzu."""
    print("\n--- SCHRITT 1: Extrahiere Text aus allen PDF-Seiten ---")
    texts = []
    total = len(df)
    for index, row in df.iterrows():
        print(f"  Verarbeite Text von Seite {index + 1}/{total}...", end='\r')
        try:
            with fitz.open(row['page_pdf_path']) as doc:
                page = doc.load_page(0)
                texts.append(page.get_text("text"))
        except Exception as e:
            print(f"  Fehler beim Lesen von {row['page_pdf_path']}: {e}")
            texts.append("")
    df['extracted_text'] = texts
    print(f"\nText-Extraktion für {total} Seiten abgeschlossen.")
    return df

def step2_classify_text(df):
    """Klassifiziert den extrahierten Text mit einem LLM auf Alkohol-Stichworte."""
    print("\n--- SCHRITT 2: Klassifiziere Texte auf Alkohol-Stichworte ---")
    text_prompt_template = load_prompt(TEXT_PROMPT_PATH)
    if not text_prompt_template: return df

    flags = []
    total = len(df)
    for index, row in df.iterrows():
        print(f"  Klassifiziere Text {index + 1}/{total}...", end='\r')
        text = row['extracted_text']
        if pd.isna(text) or len(text.strip()) < 10:
            flags.append(0)
            continue
        
        prompt = text_prompt_template.replace("{page_text}", text)
        result = call_ollama_api(prompt, TEXT_MODEL)
        flags.append(result.get('flag', 0)) # Standardwert 0 bei Fehler
        time.sleep(0.5) # Kurze Pause

    df['alc_keyword_flag'] = flags
    flagged_count = df['alc_keyword_flag'].sum()
    print(f"\nText-Klassifizierung abgeschlossen. {flagged_count} von {total} Seiten wurden als potenziell relevant markiert.")
    return df

def step3_annotate_images(df):
    """Annotiert die markierten Seiten mit einem multimodalen LLM."""
    print("\n--- SCHRITT 3: Annotiere markierte Seiten mit multimodalem LLM ---")
    image_prompt = load_prompt(IMAGE_PROMPT_PATH)
    if not image_prompt: return df

    # Filtere nur die Zeilen, die annotiert werden müssen
    df_to_process = df[df['alc_keyword_flag'] == 1].copy()
    
    if df_to_process.empty:
        print("Keine Seiten zur Bild-Annotation markiert. Workflow beendet.")
        return df

    total_to_process = len(df_to_process)
    print(f"Starte Bild-Annotation für {total_to_process} Seiten...")

    # Initialisiere die Annotationsspalten im Haupt-DataFrame mit einem Standardwert
    annotation_cols = ['alc', 'product', 'reduc', 'prod_pp', 'prod_pp_alc']
    for col in annotation_cols:
        if col not in df.columns:
            df[col] = pd.NA

    for i, (index, row) in enumerate(df_to_process.iterrows()):
        print(f"  Annotiere Bild {i + 1}/{total_to_process}: {os.path.basename(row['page_pdf_path'])}")
        
        # Bild "on-the-fly" rendern
        try:
            with fitz.open(row['page_pdf_path']) as doc:
                page = doc.load_page(0)
                pix = page.get_pixmap(dpi=IMAGE_DPI, colorspace=fitz.csGRAY if IMAGE_GRAYSCALE else fitz.csRGB)
                img = Image.frombytes("L" if IMAGE_GRAYSCALE else "RGB", [pix.width, pix.height], pix.samples)
                
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=IMAGE_QUALITY)
                image_bytes = buffer.getvalue()
        except Exception as e:
            print(f"    -> Fehler beim Rendern des Bildes: {e}")
            continue

        # Annotation durchführen
        annotation = call_ollama_api(image_prompt, IMAGE_MODEL, image_bytes=image_bytes)

        # Ergebnisse direkt in den Haupt-DataFrame schreiben
        if not annotation.get("error"):
            for col in annotation_cols:
                df.loc[index, col] = annotation.get(col, pd.NA)
        else:
             print(f"    -> Fehler bei der Annotation vom LLM erhalten.")

    print("Bild-Annotation abgeschlossen.")
    return df

# ==============================================================================
# --- HAUPTSKRIPT ---
# ==============================================================================

if __name__ == "__main__":
    try:
        df = pd.read_csv(BASE_CSV_FILE)
    except FileNotFoundError:
        print(f"FATALER FEHLER: Eingabe-CSV-Datei nicht gefunden: {BASE_CSV_FILE}")
        exit()

    print("Starte Annotations-Workflow...")
    
    if TEST_RUN:
        print(f"!!! ACHTUNG: Führe einen TESTLAUF mit den ersten {TEST_RUN_SIZE} Seiten durch. !!!")
        df = df.head(TEST_RUN_SIZE).copy()

    # Führe die Schritte nacheinander aus
    df_step1 = step1_extract_text(df)
    df_step2 = step2_classify_text(df_step1)
    df_final = step3_annotate_images(df_step2)

    # Speichere das Endergebnis
    df_final.to_csv(FINAL_OUTPUT_CSV, index=False, encoding='utf-8-sig')
    
    print(f"\n==========================================================")
    print(f"Workflow abgeschlossen! Das finale, annotierte CSV wurde gespeichert unter:")
    print(FINAL_OUTPUT_CSV)
    print(f"==========================================================")