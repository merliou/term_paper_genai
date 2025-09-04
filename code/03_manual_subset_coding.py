import pandas as pd
import subprocess
import os
import time

# Pfad zu deiner CSV-Datei
csv_file = 'subsets/subset_1.csv'

# Lade das CSV in einen Pandas DataFrame
df = pd.read_csv(csv_file)

print(f"Starte die sequenzielle Anzeige von {len(df)} PDF-Seiten.")
print("Drücke Enter, um die nächste PDF zu öffnen. Drücke 'q' und Enter, um zu beenden.")

for index, row in df.iterrows():
    pdf_path = row['page_pdf_path']
    supermarket = row['supermarket']
    date = row['date']
    page_number = row['page_number']

    print(f"\n--- Anzeigen von: {supermarket}, Datum: {date}, Seite: {page_number} ({pdf_path}) ---")

    # Überprüfen, ob der Pfad existiert
    if not os.path.exists(pdf_path):
        print(f"Fehler: PDF-Datei nicht gefunden unter {pdf_path}. Überspringe diese Datei.")
        input("Drücke Enter, um fortzufahren...") # Warte auf Bestätigung für den Fehler
        continue

    # Öffne die PDF-Datei
    try:
        # Dies öffnet die PDF-Datei mit dem Standard-PDF-Viewer deines Systems
        if os.name == 'posix':  # macOS oder Linux
            # Für Linux (häufig):
            subprocess.Popen(['xdg-open', pdf_path])
            # Für macOS (falls xdg-open nicht verfügbar oder bevorzugt):
            # subprocess.Popen(['open', pdf_path])
        elif os.name == 'nt':  # Windows
            os.startfile(pdf_path)
        else:
            print(f"Unbekanntes Betriebssystem. Kann {pdf_path} nicht automatisch öffnen.")
            input("Drücke Enter, um fortzufahren...")
            continue

    except Exception as e:
        print(f"Fehler beim Öffnen der PDF {pdf_path}: {e}")
        input("Drücke Enter, um fortzufahren...")
        continue

    # Warte auf Benutzereingabe, bevor die nächste PDF geöffnet wird
    # Dadurch bleibt die aktuelle PDF geöffnet, bis du bereit bist für die nächste
    user_input = input("Drücke Enter für nächste PDF, oder 'q' zum Beenden: ").strip().lower()

    if user_input == 'q':
        print("Sequenzielle Anzeige beendet.")
        break

print("Alle PDFs wurden angezeigt oder die Anzeige wurde beendet.")
print("Denke daran, deine Kodierungen in Excel zu speichern!")