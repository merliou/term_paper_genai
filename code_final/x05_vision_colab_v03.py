# 00 mount drive
from google.colab import drive
drive.mount('/content/drive')

## 00 install missing dependencies
!pip install ollama pandas pymupdf pillow tqdm

# 00 set up ollama in shell
!curl -fsSL https://ollama.com/install.sh | sh

# 00 run ollama so that API is online
import subprocess
import time

# Start Ollama service in the background
process = subprocess.Popen(['ollama', 'serve'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Give Ollama a moment to start
time.sleep(5)

# Verify Ollama is running by listing models
try:
    result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, check=True, timeout=30)
    print("Ollama is running and accessible:")
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print(f"Error verifying Ollama: {e}")
    print(f"Stderr: {e.stderr}")
    # Try to read from the ollama serve process stderr to see if there are startup errors
    try:
        startup_errors = process.communicate(timeout=5)[1].decode('utf-8')
        print(f"Ollama serve startup errors: {startup_errors}")
    except subprocess.TimeoutExpired:
        print("Could not retrieve startup errors from ollama serve process.")
    except Exception as communicate_error:
        print(f"Error communicating with ollama serve process: {communicate_error}")
except FileNotFoundError:
    print("Ollama command not found. Make sure it's in the PATH.")
except subprocess.TimeoutExpired:
    print("Ollama list command timed out.")

# 00 download models
import subprocess

try:
    result = subprocess.run(
        ['ollama', 'pull', 'llama3.2-vision:11b-instruct-fp16'],
        capture_output=True,
        text=True,
        check=True
    )
    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr)
except subprocess.CalledProcessError as e:
    print(f"Error pulling model: {e}")
    print(f"STDOUT: {e.stdout}")
    print(f"STDERR: {e.stderr}")
except FileNotFoundError:
    print("Ollama command not found. Make sure it's in the PATH.")


########################################################################
# 01 ACTUAL SCRIPT
########################################################################
# import necessary stuff
import pandas as pd
import os
import time
import ollama  # NEU: Ollama-Bibliothek importiert
import fitz  # PyMuPDF
from PIL import Image
from io import BytesIO
import json
from tqdm import tqdm
import glob


# ========== anaconda-project run python x06_ollama_approach_annotation_run_v01.py ==========
# --- KONFIGURATION ---
# ===========================================================================================

BASE_FOLDER = '/content/drive/MyDrive/term_paper_genai'

SUBSET_INPUT_FOLDER = os.path.join(BASE_FOLDER, 'subsets_for_annotation')
ANNOTATION_OUTPUT_FOLDER = os.path.join(BASE_FOLDER, 'annotations_ollama_llava:13b')
PROMPT_FILE_PATH = os.path.join(BASE_FOLDER, "prompts/03_api_annotation_prompt_v01.md")


# GEÄNDERT: Modell- und Host-Konfiguration für Ollama
OLLAMA_MODEL = "llama3.2-vision:11b-instruct-fp16"
# OLLAMA_HOST = "http://localhost:11434"

# NEU: Seed für reproduzierbare Ergebnisse setzen.
# Ändern Sie die Zahl, um andere (aber konsistente) Ergebnisse zu erhalten.
OLLAMA_SEED = 42

# GEÄNDERT: Generation Config für Ollama (Parameter können abweichen)
# 'temperature' funktioniert bei Ollama genauso.
# Der Seed wird hinzugefügt, um die Reproduzierbarkeit der Ergebnisse zu gewährleisten.
GENERATION_CONFIG = {
    "temperature": 0,
    "seed": OLLAMA_SEED,
}

# Bild-Rendering-Einstellungen (bleiben unverändert)
IMAGE_DPI = 96
IMAGE_GRAYSCALE = True
IMAGE_QUALITY = 75

# Spalten, die durch die Annotation befüllt werden sollen (bleiben unverändert)
ANNOTATION_COLS = [
    'alc',
    'product',
    'warning',
    'reduc',
    'child',
    'prod_pp',
    'prod_alc'
]
ERROR_COL = 'ollama_error' # GEÄNDERT: Spaltenname für Fehler angepasst


# ==============================================================================
# --- HILFSFUNKTIONEN ---
# ==============================================================================

# ENTFERNT: setup_api_key() wird für lokales Ollama nicht benötigt.

def prepare_all_csv_files(folder_path):
    """
    Stellt sicher, dass alle CSV-Dateien im angegebenen Ordner die für die
    Annotation notwendigen Spalten enthalten. (Logik unverändert)
    """
    print(f"\nÜberprüfe und vorbereite alle CSV-Dateien im Ordner '{folder_path}'...")
    csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
    if not csv_files:
        print(f"WARNUNG: Keine CSV-Dateien im Ordner '{folder_path}' gefunden.")
        return

    columns_to_check = ANNOTATION_COLS + [ERROR_COL] # GEÄNDERT: nutzt neuen Fehlerspalten-Namen

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
    """Lädt einen Prompt-Text aus einer angegebenen Datei. (unverändert)"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"FATALER FEHLER: Prompt-Datei nicht gefunden unter: {file_path}")
        return None

# ==============================================================================
# --- HAUPTFUNKTIONEN ---
# ==============================================================================

# GEÄNDERT: Komplette Funktion zum Aufruf von Ollama statt Gemini
def annotate_page_with_ollama(pdf_path, client, model_name, prompt_content, generation_options):
    """
    Rendert eine PDF-Seite als Bild, sendet sie an ein Ollama Vision-Modell
    und gibt das Ergebnis-JSON zurück.
    """
    # 1. PDF-Seite als Bild rendern (Logik ist identisch)
    try:
        with fitz.open(pdf_path) as doc:
            page = doc.load_page(0)
            pix = page.get_pixmap(dpi=IMAGE_DPI, colorspace=fitz.csGRAY if IMAGE_GRAYSCALE else fitz.csRGB)
            img = Image.frombytes("L" if IMAGE_GRAYSCALE else "RGB", [pix.width, pix.height], pix.samples)

            buffer = BytesIO()
            img.save(buffer, format="JPEG", quality=IMAGE_QUALITY)
            image_bytes = buffer.getvalue() # Wir benötigen die Bytes des Bildes

    except Exception as e:
        return {"error": f"Image rendering failed: {e}"}

    # 2. Ollama API aufrufen
    try:
        response = client.chat(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': prompt_content,
                    'images': [image_bytes] # Bild-Bytes direkt übergeben
                }
            ],
            options=generation_options, # Optionen wie temperature und seed übergeben
            format="json" # NEU: Ollama anweisen, direkt ein JSON-Objekt auszugeben
        )

        # Die Antwort von Ollama ist bereits ein JSON-String, wenn format="json" verwendet wird.
        # Wir müssen ihn nur noch in ein Python-Dictionary umwandeln.
        # Der JSON-String befindet sich in response['message']['content']
        cleaned_response = response['message']['content']
        return json.loads(cleaned_response)

    except Exception as e:
        return {"error": f"Ollama API call failed: {e}"}


def process_subset(input_csv_path, output_csv_path, client, model, prompt, config):
    """
    Führt den Annotations-Workflow für eine einzelne Subset-CSV-Datei aus.
    """
    try:
        df = pd.read_csv(input_csv_path)
        print(f"\nStarte Annotation für '{os.path.basename(input_csv_path)}' ({len(df)} Seiten).")
    except FileNotFoundError:
        print(f"FEHLER: Eingabedatei nicht gefunden: {input_csv_path}")
        return

    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Annotiere {os.path.basename(input_csv_path)}"):
        pdf_path = row['page_pdf_path']

        # Pfade für Colab anpassen, falls sie relativ sind
        if not os.path.isabs(pdf_path):
             pdf_path = os.path.join(BASE_FOLDER, pdf_path)


        if pd.notna(row.get(ANNOTATION_COLS[0])) and pd.isna(row.get(ERROR_COL)):
             continue

        if not os.path.exists(pdf_path):
            df.loc[index, ERROR_COL] = f"File not found: {pdf_path}"
            continue

        # GEÄNDERT: Ruft die neue Ollama-Funktion auf
        result = annotate_page_with_ollama(pdf_path, client, model, prompt, config)

        if "error" in result:
            df.loc[index, ERROR_COL] = result["error"]
        else:
            df.loc[index, ERROR_COL] = None
            for col in ANNOTATION_COLS:
                df.loc[index, col] = result.get(col, pd.NA)

        if (index + 1) % 50 == 0:
            df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    print(f"-> Annotation für '{os.path.basename(input_csv_path)}' abgeschlossen.")
    print(f"-> Ergebnisse gespeichert in: {output_csv_path}")

# ==============================================================================
# --- HAUPTSKRIPT (STEUERUNG) ---
# ==============================================================================
if __name__ == "__main__":
    print("==========================================================")
    print("=== Prospekt-Annotation mit Ollama gestartet           ===")
    print("==========================================================")

    # ENTFERNT: API-Key-Konfiguration

    # Lade den Prompt
    prompt_content = load_prompt_from_file(PROMPT_FILE_PATH)
    if prompt_content is None:
        exit()
    print("Prompt erfolgreich geladen.")

    prepare_all_csv_files(SUBSET_INPUT_FOLDER)
    os.makedirs(ANNOTATION_OUTPUT_FOLDER, exist_ok=True)

    # NEU: Initialisiere den Ollama-Client
    try:
        # Falls Sie einen OLLAMA_HOST konfiguriert haben:
        # client = ollama.Client(host=OLLAMA_HOST)
        client = ollama.Client()
        # Ping den Server, um sicherzustellen, dass er erreichbar ist
        client.list()
        print(f"Erfolgreich mit Ollama verbunden. Standardmodell: {OLLAMA_MODEL}")
    except Exception as e:
        print(f"FATALER FEHLER: Konnte keine Verbindung zu Ollama herstellen.")
        print("Stellen Sie sicher, dass Ollama läuft (entweder lokal oder in Colab).")
        print(f"Fehlerdetails: {e}")
        exit()


    # Der Rest der Logik mit der Modus-Auswahl bleibt identisch.
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
            test_filename = input("Geben Sie den Dateinamen des Test-Subsets an (z.B. subset_1.csv): ")
            input_path = os.path.join(SUBSET_INPUT_FOLDER, test_filename)
            output_filename = test_filename.replace('.csv', '_annotated_ollama.csv')
            output_path = os.path.join(ANNOTATION_OUTPUT_FOLDER, output_filename)

            if not os.path.exists(input_path):
                print(f"FEHLER: Die Testdatei '{input_path}' wurde nicht gefunden.")
                continue

            print(f"\n--- Starte TESTLAUF für {test_filename} ---")
            process_subset(input_path, output_path, client, OLLAMA_MODEL, prompt_content, GENERATION_CONFIG)
            break

        elif mode == '2':
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
                output_filename = os.path.basename(subset_path).replace('.csv', '_annotated_ollama.csv')
                output_path = os.path.join(ANNOTATION_OUTPUT_FOLDER, output_filename)
                process_subset(subset_path, output_path, client, OLLAMA_MODEL, prompt_content, GENERATION_CONFIG)
            break

        elif mode.lower() == 'x':
            print("Skript beendet.")
            break
        else:
            print("Ungültige Eingabe. Bitte wählen Sie 1, 2 oder x.")

    print(f"\n==========================================================")
    print(f"Workflow abgeschlossen!")
    print(f"==========================================================")