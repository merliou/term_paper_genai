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
import glob # Hinzugefügt, um einfach nach Dateien zu suchen

# ==============================================================================
# --- KONFIGURATION ---
# ==============================================================================

# Lädt die Umgebungsvariablen aus der .env Datei (muss im selben Ordner liegen)
load_dotenv()

# Ordner für die Eingabe- und Ausgabedateien
SUBSET_INPUT_FOLDER = 'subsets_for_annotation'
ANNOTATION_OUTPUT_FOLDER = 'annotations_api_gemini_2.0_flash'

PROMPT_FILE_PATH = "term_paper_genai/prompts/03_api_annotation_prompt_v01.md"
GEMINI_MODEL = "gemini-2.0-flash"
# Setzen der Temperatur auf 0 für maximale Reproduzierbarkeit der Ergebnisse.
GENERATION_CONFIG = {
    "temperature": 0,
}

# Bild-Rendering-Einstellungen (Kompromiss zwischen Qualität und Kosten)
IMAGE_DPI = 96      # Niedrigere DPI = kleinere Dateigröße & Kosten
IMAGE_GRAYSCALE = True # Graustufen sind für Texterkennung oft ausreichend
IMAGE_QUALITY = 75  # JPEG-Qualität

# Spalten, die durch die Annotation befüllt werden sollen
ANNOTATION_COLS = [
    'alc',
    'product',
    'warning',
    'reduc',
    'child',
    'prod_pp',
    'prod_alc'
]
# Spalte für Fehlermeldungen
ERROR_COL = 'gemini_error'

# ==============================================================================
# --- HILFSFUNKTIONEN ---
# ==============================================================================

def setup_api_key():
    """Konfiguriert den Google API Key und beendet das Skript bei einem Fehler."""
    try:
        GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
        if GOOGLE_API_KEY is None:
            raise ValueError("GOOGLE_API_KEY nicht in der .env Datei gefunden.")
        genai.configure(api_key=GOOGLE_API_KEY)
        print("API-Key erfolgreich konfiguriert.")
    except Exception as e:
        print(f"FATALER FEHLER: {e}")
        print("Stellen Sie sicher, dass eine .env Datei mit Ihrem GOOGLE_API_KEY im selben Ordner liegt.")
        print("Anleitung für den API-Key: https://aistudio.google.com/app/apikey")
        exit()

def prepare_all_csv_files(folder_path):
    """
    Stellt sicher, dass alle CSV-Dateien im angegebenen Ordner die für die
    Annotation notwendigen Spalten enthalten.
    """
    print(f"\nÜberprüfe und vorbereite alle CSV-Dateien im Ordner '{folder_path}'...")
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    if not csv_files:
        print(f"WARNUNG: Keine CSV-Dateien im Ordner '{folder_path}' gefunden.")
        return

    columns_to_check = ANNOTATION_COLS + [ERROR_COL]

    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            updated = False
            for col in columns_to_check:
                if col not in df.columns:
                    df[col] = None
                    updated = True

            if updated:
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
                print(f" -> '{os.path.basename(file_path)}' wurde aktualisiert.")
        except Exception as e:
            print(f" -> Fehler beim Verarbeiten von '{os.path.basename(file_path)}': {e}")
    print("Vorbereitung der CSV-Dateien abgeschlossen.")


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

def annotate_page_with_gemini(pdf_path, model, prompt_content, generation_config):
    """
    Rendert eine PDF-Seite als Bild, sendet sie an Gemini und gibt das Ergebnis-JSON zurück.
    """
    # 1. PDF-Seite als Bild rendern
    try:
        with fitz.open(pdf_path) as doc:
            page = doc.load_page(0) # Annahme: jede PDF-Datei hat nur eine Seite
            pix = page.get_pixmap(
                dpi=IMAGE_DPI,
                colorspace=fitz.csGRAY if IMAGE_GRAYSCALE else fitz.csRGB
            )
            img = Image.frombytes("L" if IMAGE_GRAYSCALE else "RGB", [pix.width, pix.height], pix.samples)

            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=IMAGE_QUALITY)
            image_for_api = Image.open(buffer)

    except Exception as e:
        return {"error": f"Image rendering failed: {e}"}

    # 2. Gemini API aufrufen
    try:
        # --- ANGEPASST: Übergabe der generation_config mit temperature=0 ---
        response = model.generate_content(
            [prompt_content, image_for_api],
            generation_config=generation_config
        )

        # Bereinigen der Antwort, um nur das JSON zu extrahieren
        cleaned_response = response.text.strip().replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned_response)

    except Exception as e:
        return {"error": f"Gemini API call failed: {e}"}


def process_subset(input_csv_path, output_csv_path, model, prompt, config):
    """
    Führt den Annotations-Workflow für eine einzelne Subset-CSV-Datei aus.
    """
    try:
        df = pd.read_csv(input_csv_path)
        print(f"\nStarte Annotation für '{os.path.basename(input_csv_path)}' ({len(df)} Seiten).")
    except FileNotFoundError:
        print(f"FEHLER: Eingabedatei nicht gefunden: {input_csv_path}")
        return

    # tqdm sorgt für eine Fortschrittsanzeige
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Annotiere {os.path.basename(input_csv_path)}"):
        pdf_path = row['page_pdf_path']

        # Überspringe bereits erfolgreich annotierte Zeilen (optional, aber nützlich bei Wiederaufnahme)
        if pd.notna(row.get(ANNOTATION_COLS[0])) and pd.isna(row.get(ERROR_COL)):
             continue

        if not os.path.exists(pdf_path):
            df.loc[index, ERROR_COL] = "File not found"
            continue

        # API-Aufruf durchführen
        result = annotate_page_with_gemini(pdf_path, model, prompt, config)

        # Ergebnisse in den DataFrame schreiben
        if "error" in result:
            df.loc[index, ERROR_COL] = result["error"]
        else:
            df.loc[index, ERROR_COL] = None # Fehler löschen, falls zuvor einer bestand
            for col in ANNOTATION_COLS:
                df.loc[index, col] = result.get(col, pd.NA)

        # Optionale Zwischenspeicherung
        if (index + 1) % 50 == 0:
            df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

    # Finale Speicherung der Ergebnisse für diese Datei
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"-> Annotation für '{os.path.basename(input_csv_path)}' abgeschlossen.")
    print(f"-> Ergebnisse gespeichert in: {output_csv_path}")

# ==============================================================================
# --- HAUPTSKRIPT (STEUERUNG) ---
# ==============================================================================
if __name__ == "__main__":
    print("==========================================================")
    print("=== Prospekt-Annotation mit Gemini 2.0 Flash gestartet ===")
    print("==========================================================")

    # API-Key konfigurieren
    setup_api_key()

    # Lade den Prompt
    prompt_content = load_prompt_from_file(PROMPT_FILE_PATH)
    if prompt_content is None:
        exit()
    print("Prompt erfolgreich geladen.")

    # Bereite alle CSVs im Input-Ordner vor (füge Spalten hinzu, falls nötig)
    prepare_all_csv_files(SUBSET_INPUT_FOLDER)

    # Erstelle den Ausgabeordner, falls er nicht existiert
    os.makedirs(ANNOTATION_OUTPUT_FOLDER, exist_ok=True)

    # Initialisiere das Gemini-Modell
    model = genai.GenerativeModel(GEMINI_MODEL)

    # --- NEU: Auswahl des Ausführungsmodus ---
    while True:
        print("\n--- Modus auswählen ---")
        mode = input(
            "Wählen Sie eine Option:\n"
            "  1: Testlauf (annotiert eine einzelne, anzugebende Datei)\n"
            "  2: Vollständiger Lauf (annotiert ALLE restlichen Dateien im Input-Ordner)\n"
            "  x: Beenden\n"
            "Ihre Wahl: "
        )

        if mode == '1':
            # --- TESTLAUF ---
            test_filename = input("Geben Sie den Dateinamen des Test-Subsets an (z.B. subset_1.csv): ")
            input_path = os.path.join(SUBSET_INPUT_FOLDER, test_filename)
            output_filename = test_filename.replace('.csv', '_annotated.csv')
            output_path = os.path.join(ANNOTATION_OUTPUT_FOLDER, output_filename)

            if not os.path.exists(input_path):
                print(f"FEHLER: Die Testdatei '{input_path}' wurde nicht gefunden.")
                continue

            print(f"\n--- Starte TESTLAUF für {test_filename} ---")
            process_subset(input_path, output_path, model, prompt_content, GENERATION_CONFIG)
            break

        elif mode == '2':
            # --- VOLLSTÄNDIGER LAUF ---
            print("\n--- Starte VOLLSTÄNDIGEN LAUF ---")
            confirm = input(f"WARNUNG: Dies wird alle CSV-Dateien im Ordner '{SUBSET_INPUT_FOLDER}' verarbeiten. Fortfahren? (j/n): ")
            if confirm.lower() != 'j':
                print("Abgebrochen.")
                continue

            all_subsets = sorted(glob.glob(os.path.join(SUBSET_INPUT_FOLDER, '*.csv')))
            if not all_subsets:
                 print("Keine Subsets im Input-Ordner gefunden. Beende.")
                 break
            
            print(f"Gefunden: {len(all_subsets)} Subset-Dateien.")
            for i, subset_path in enumerate(all_subsets):
                print(f"\n--- Verarbeite Datei {i+1}/{len(all_subsets)} ---")
                output_filename = os.path.basename(subset_path).replace('.csv', '_annotated.csv')
                output_path = os.path.join(ANNOTATION_OUTPUT_FOLDER, output_filename)
                process_subset(subset_path, output_path, model, prompt_content, GENERATION_CONFIG)
            break

        elif mode.lower() == 'x':
            print("Skript beendet.")
            break
        else:
            print("Ungültige Eingabe. Bitte wählen Sie 1, 2 oder x.")

    print(f"\n==========================================================")
    print(f"Workflow abgeschlossen!")
    print(f"==========================================================")