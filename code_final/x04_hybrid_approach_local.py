import pandas as pd
import os
import time
import base64
import json
import requests
import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO

# Ollama Server im Terminal starten! 
# ollama run gemma3:4b
# ollama run qwen2.5vl:7b
# ollama run qwen2.5vl:3b


# --- KONFIGURATION ---
SUBSET_TO_PROCESS = 'subsets/subset_1.csv'
OUTPUT_CSV = 'annotations/subset_1_annotations_hybrid.csv'
TEMP_IMAGE_PATH = 'temp_image.jpg' # Temporärer Speicherort für Bilder

# LLM Konfiguration
OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
# WICHTIG: Wählen Sie ein kleines, quantisiertes Modell!
MODEL_NAME = "qwen2.5vl:3b" # z.B. Mistral-basiertes Llava mit 4-bit Quantisierung llava:7b-v1.6-mistral-q4_K_M

# Schwellenwert: Wie viele Zeichen Text müssen mindestens auf der Seite sein,
# um eine text-basierte Annotation überhaupt zu versuchen?
TEXT_MIN_CHARS = 50

# --- HILFSFUNKTIONEN ---

def load_prompt(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"FEHLER: Prompt-Datei nicht gefunden: {file_path}")
        return None

def extract_text_from_page(pdf_path):
    """Extrahiert reinen Text von einer PDF-Seite."""
    try:
        with fitz.open(pdf_path) as doc:
            page = doc.load_page(0)
            return page.get_text("text")
    except Exception as e:
        print(f"Fehler beim Extrahieren von Text aus {pdf_path}: {e}")
        return ""

def render_page_as_image(pdf_path, dpi=100, grayscale=True):
    """Rendert eine PDF-Seite bei Bedarf als optimiertes Bild im Speicher."""
    try:
        with fitz.open(pdf_path) as doc:
            page = doc.load_page(0)
            pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY if grayscale else fitz.csRGB)
            img = Image.frombytes("L" if grayscale else "RGB", [pix.width, pix.height], pix.samples)
            
            # Bild in Bytes umwandeln, um es direkt senden zu können
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=80)
            return buffer.getvalue()
    except Exception as e:
        print(f"Fehler beim Rendern des Bildes von {pdf_path}: {e}")
        return None

def call_ollama_api(prompt, image_bytes=None):
    """Sendet eine Anfrage an die Ollama API."""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "format": "json",
        "stream": False
    }
    if image_bytes:
        payload["images"] = [base64.b64encode(image_bytes).decode('utf-8')]

    try:
        response = requests.post(OLLAMA_ENDPOINT, json=payload, timeout=480)
        response.raise_for_status()
        response_text = response.json().get('response', '{}')
        return json.loads(response_text)
    except (requests.RequestException, json.JSONDecodeError) as e:
        print(f"API-Fehler oder JSON-Decode-Fehler: {e}")
        return {"error": str(e)}

# --- HAUPT-WORKFLOW ---

def main():
    # Lade die Prompts
    text_prompt_template = load_prompt("term_paper_genai/prompts/text_annotation_prompt.txt")
    image_prompt = load_prompt("term_paper_genai/prompts/image_annotation_prompt.txt")
    if not text_prompt_template or not image_prompt:
        return

    # Lade das Subset
    df = pd.read_csv(SUBSET_TO_PROCESS)
    all_annotations = []
    
    # Erstelle Ausgabeordner, falls nicht vorhanden
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    print(f"Starte hybride Annotation für {len(df)} Seiten aus {SUBSET_TO_PROCESS}...")

    for index, row in df.iterrows():
        pdf_path = row['page_pdf_path']
        print(f"\n[{index + 1}/{len(df)}] Verarbeite: {os.path.basename(pdf_path)}")
        
        # --- VERSUCH 1: TEXT-BASIERT ---
        page_text = extract_text_from_page(pdf_path)
        annotation = None

        if len(page_text) > TEXT_MIN_CHARS:
            print("-> Versuche schnelle, text-basierte Annotation...")
            prompt = text_prompt_template.replace("{page_text}", page_text)
            annotation = call_ollama_api(prompt)
            
            # Überprüfe, ob das LLM mehr Infos (ein Bild) braucht
            if annotation.get("error") == "insufficient text":
                print("-> Text nicht ausreichend, wechsle zur Bild-Annotation.")
                annotation = None # Setze zurück, um Bild-Modus zu erzwingen
            else:
                 print("-> Text-Annotation erfolgreich!")

        else:
            print("-> Zu wenig Text auf der Seite, starte direkt mit Bild-Annotation.")

        # --- VERSUCH 2: BILD-BASIERT (FALLBACK) ---
        if not annotation:
            print("-> Rendere Bild und starte multimodale Annotation...")
            image_bytes = render_page_as_image(pdf_path)
            if image_bytes:
                annotation = call_ollama_api(image_prompt, image_bytes=image_bytes)
                if not annotation.get("error"):
                    print("-> Bild-Annotation erfolgreich!")
                else:
                    print(f"-> Fehler bei Bild-Annotation: {annotation.get('error')}")
            else: # Falls das Rendern fehlschlägt
                annotation = {key: 98 for key in ["alc", "product", "child", "reduc", "prod_pp", "prod_pp_alc"]}


        # Füge Metadaten hinzu und speichere das Ergebnis
        if annotation and not annotation.get("error"):
            annotation['filename'] = os.path.basename(pdf_path)
            annotation['annotation_method'] = 'text' if len(page_text) > TEXT_MIN_CHARS and 'error' not in annotation else 'image'
            all_annotations.append(annotation)
        
        time.sleep(1) # Kurze Pause, um die API nicht zu überlasten

    # Speichere alle Ergebnisse in einer CSV-Datei
    if all_annotations:
        result_df = pd.DataFrame(all_annotations)
        result_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nAnnotation abgeschlossen! {len(all_annotations)} Seiten erfolgreich verarbeitet.")
        print(f"Ergebnisse gespeichert in: {OUTPUT_CSV}")
    else:
        print("\nKeine Seiten konnten annotiert werden.")


if __name__ == "__main__":
    main()