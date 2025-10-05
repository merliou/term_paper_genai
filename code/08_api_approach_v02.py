########################################################################
# import necessary stuff
import pandas as pd
import os
import time
import google.generativeai as genai
import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO
import json
from tqdm import tqdm # Für eine schöne Fortschrittsanzeige
from dotenv import load_dotenv



########################## PREP ########################################

# add necessary columns (according to codebook) to csv (if not existing)
new_columns = [
    'alc',
    'product',
    'warning',
    'reduc',
    'child',
    'prod_pp',
    'prod_alc'
]

# Der Ordner, in dem deine CSV-Dateien liegen
folder_path = 'subsets_for_annotation' # Pass den Ordnernamen an, falls nötig

# Gehe durch jede Datei im angegebenen Ordner
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)

        # Lade die CSV-Datei
        df = pd.read_csv(file_path)

        # Füge jede neue Spalte hinzu, falls sie noch nicht existiert
        for col in new_columns:
            if col not in df.columns:
                # Fülle die neue Spalte mit einem Standardwert, z.B. None (leer)
                # oder 0, je nachdem, was für dich besser ist.
                df[col] = None

        # Speichere die aktualisierte Datei (überschreibe die alte)
        df.to_csv(file_path, index=False)
        print(f"'{filename}' wurde erfolgreich aktualisiert.")

print("Alle CSV-Dateien wurden verarbeitet.")


##########################################################

# ==============================================================================
# --- KONFIGURATION ---
# ==============================================================================

load_dotenv() # Lädt die Variablen aus der .env Datei
try:
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    if GOOGLE_API_KEY is None:
        raise ValueError("GOOGLE_API_KEY nicht in .env Datei gefunden.")
    genai.configure(api_key=GOOGLE_API_KEY)
    print("API-Key erfolgreich konfiguriert.")
except Exception as e:
    print(f"FATALER FEHLER: {e}")
    print("Anleitung: https://aistudio.google.com/app/apikey")
    exit()


# Eingabedatei mit den Pfaden zu den PDFs (z.B. 'subsets/subset_2.csv')
BASE_CSV_FILE = 'subsets/subset_23.csv'
# Ausgabedatei für die Ergebnisse
OUTPUT_CSV_FILE = 'annotations/subset_23_gemini_1.5_pro_annotated.csv'

# Gemini-Modell
GEMINI_MODEL = "gemini-1.5-pro" # Empfehlung: 1.5 Pro ist oft schneller und günstiger für Bild-Tasks

# Bild-Rendering-Einstellungen (Kompromiss zwischen Qualität und Kosten)
IMAGE_DPI = 96      # Niedrigere DPI = kleinere Dateigröße & Kosten
IMAGE_GRAYSCALE = True # Graustufen sind für Texterkennung oft ausreichend
IMAGE_QUALITY = 75  # JPEG-Qualität

PROMPT_FILE_PATH = "term_paper_genai/prompts/03_api_annotation_prompt_v01.md"


# ==============================================================================
# --- HILFSFUNKTIONEN ---
# ==============================================================================

def load_prompt_from_file(file_path):
    """Lädt einen Prompt-Text aus einer angegebenen Datei."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"FATALER FEHLER: Prompt-Datei nicht gefunden unter: {file_path}")
        return None


# ==============================================================================
# --- HAUPTFUNKTIONEN ---
# ==============================================================================

def annotate_page_with_gemini(pdf_path, prompt_content):
    """
    Rendert eine PDF-Seite als Bild, sendet sie an Gemini und gibt das Ergebnis-JSON zurück.
    """
    try:
        # 1. PDF-Seite als Bild rendern
        with fitz.open(pdf_path) as doc:
            page = doc.load_page(0) # Annahme: jede PDF-Datei hat nur eine Seite
            pix = page.get_pixmap(
                dpi=IMAGE_DPI,
                colorspace=fitz.csGRAY if IMAGE_GRAYSCALE else fitz.csRGB
            )
            img = Image.frombytes("L" if IMAGE_GRAYSCALE else "RGB", [pix.width, pix.height], pix.samples)

            # Bild in Bytes umwandeln
            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=IMAGE_QUALITY)
            image_bytes = buffer.getvalue()

            # PIL Image-Objekt für die API erstellen
            image_for_api = Image.open(BytesIO(image_bytes))

    except Exception as e:
        print(f"  -> Fehler beim Rendern von {os.path.basename(pdf_path)}: {e}")
        return {"error": f"Image rendering failed: {e}"}

    # 2. Gemini API aufrufen
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content([prompt_content, image_for_api])

        # Bereinigen der Antwort, um nur das JSON zu extrahieren
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()

        return json.loads(cleaned_response)

    except Exception as e:
        print(f"  -> Fehler bei der Gemini API für {os.path.basename(pdf_path)}: {e}")
        return {"error": f"Gemini API call failed: {e}"}


# ==============================================================================
# --- HAUPTSKRIPT ---
# ==============================================================================
if __name__ == "__main__":
    print("Starte Workflow: Prospekt-Annotation mit Gemini 1.5 Pro.")

    # Lade den Prompt
    prompt_content = load_prompt_from_file(PROMPT_FILE_PATH)
    if prompt_content is None:
        exit() # Beendet das Skript, wenn die Prompt-Datei nicht geladen werden konnte
    print("Prompt erfolgreich geladen.")

    # Eingabedatei prüfen und laden
    try:
        df = pd.read_csv(BASE_CSV_FILE)
        print(f"Basis-Datei '{BASE_CSV_FILE}' mit {len(df)} Seiten erfolgreich geladen.")
    except FileNotFoundError:
        print(f"FATALER FEHLER: Basis-CSV-Datei nicht gefunden: {BASE_CSV_FILE}")
        exit()

    # --- ÄNDERUNG ---
    # Die Liste der zu annotierenden Spalten wurde entsprechend Ihrer Anforderung aktualisiert.
    # Die Spalte 'gemini_error' wird beibehalten, um Fehler zu protokollieren.
    annotation_cols = [
        'alc',
        'product',
        'warning',
        'reduc',
        'child',
        'prod_pp',
        'prod_alc'
    ]

    # Verzeichnis für die Ausgabe erstellen, falls es nicht existiert
    output_dir = os.path.dirname(OUTPUT_CSV_FILE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Schleife über alle Zeilen im DataFrame
    # tqdm sorgt für eine Fortschrittsanzeige
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Annotiere Seiten"):
        pdf_path = row['page_pdf_path']

        if not os.path.exists(pdf_path):
            df.loc[index, 'gemini_error'] = "File not found"
            continue

        # API-Aufruf durchführen
        result = annotate_page_with_gemini(pdf_path, prompt_content)

        # Ergebnisse in den DataFrame schreiben
        if "error" in result:
            df.loc[index, 'gemini_error'] = result["error"]
        else:
            # Diese Schleife füllt nun die Werte für die korrekten Spalten aus dem API-Ergebnis.
            for col in annotation_cols:
                # result.get(col, pd.NA) sucht nach dem Spaltennamen im Ergebnis
                # und trägt pd.NA (einen leeren Wert) ein, falls der Schlüssel nicht gefunden wird.
                df.loc[index, col] = result.get(col, pd.NA)

        # Optionale Zwischenspeicherung nach X Zeilen
        if (index + 1) % 20 == 0:
            df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8-sig')

    # Finale Speicherung der Ergebnisse
    df.to_csv(OUTPUT_CSV_FILE, index=False, encoding='utf-8-sig')

    print(f"\n==========================================================")
    print(f"Workflow abgeschlossen! Die Ergebnisse wurden gespeichert in:")
    print(OUTPUT_CSV_FILE)
    print(f"==========================================================")