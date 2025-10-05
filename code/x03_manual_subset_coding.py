import pandas as pd
import subprocess
import os
import time

# --- Konfiguration ---
# Gib hier den Pfad zu dem Subset an, das du annotieren möchtest.
CSV_FILE_TO_ANNOTATE = 'subsets_for_annotation/subset_3.csv'

# Definiere die Kategorien für deinen Gold-Standard.
GOLD_STANDARD_COLUMNS = [
    "alc_gold",
    "product_gold",
    "warning_gold",
    "reduc_gold",
    "child_gold",
    "prod_pp_gold",
    "prod_alc_gold"
]

def prepare_csv_for_annotation(csv_file):
    """Fügt die Gold-Standard-Spalten zur CSV hinzu, falls sie fehlen."""
    df = pd.read_csv(csv_file)
    
    # Prüfen, ob Spalten bereits existieren
    missing_cols = [col for col in GOLD_STANDARD_COLUMNS if col not in df.columns]
    
    if missing_cols:
        print(f"Füge fehlende Spalten hinzu: {', '.join(missing_cols)}")
        for col in missing_cols:
            df[col] = pd.NA  # Mit leeren Werten initialisieren
        
        # Speichere die erweiterte CSV-Datei
        df.to_csv(csv_file, index=False)
        print("CSV-Datei wurde für die Annotation vorbereitet.")
    else:
        print("Alle Annotations-Spalten sind bereits vorhanden.")
        
    return df

def start_annotation_session(df, csv_file):
    """Startet die interaktive Annotations-Schleife."""
    
    print("\n--- ANNOTATION STARTEN ---")
    print("Anleitung:")
    print("  - Geben Sie eine Zahl von 0-99 für die jeweilige Kategorie ein.")
    print("  - Drücken Sie nur 'Enter', um den Wert auf 0 zu setzen.")
    print("\nSteuerung:")
    print("  - 'q' zum Beenden (speichert den Fortschritt)")
    print("  - 's' zum Überspringen der aktuellen Seite")
    print("  - 'b' zum Zurückgehen zur vorherigen Kategorie auf derselben Seite")
    print("-" * 30)

    for index, row in df.iterrows():
        # Prüfen, ob die Zeile bereits vollständig annotiert ist
        if not row[GOLD_STANDARD_COLUMNS].isnull().any():
            user_choice = input(f"Seite {index + 1}/{len(df)} ist bereits annotiert. Erneut bearbeiten? (j/n): ").lower()
            if user_choice != 'j':
                continue

        pdf_path = row['page_pdf_path']
        print(f"\n--- Seite {index + 1}/{len(df)}: {os.path.basename(pdf_path)} ---")

        if not os.path.exists(pdf_path):
            print(f"FEHLER: PDF nicht gefunden: {pdf_path}. Überspringe.")
            continue

        # PDF öffnen
        try:
            if os.name == 'posix':  # macOS oder Linux
                subprocess.Popen(['xdg-open', pdf_path])
            elif os.name == 'nt':  # Windows
                os.startfile(pdf_path)
        except Exception as e:
            print(f"FEHLER beim Öffnen der PDF: {e}")
            continue

        annotations = {}
        current_col_idx = 0
        
        while current_col_idx < len(GOLD_STANDARD_COLUMNS):
            col = GOLD_STANDARD_COLUMNS[current_col_idx]
            user_input = input(f"  -> {col}? ").strip().lower()

            if user_input == 'q':
                print("Session beendet. Dein Fortschritt ist gespeichert.")
                return
            elif user_input == 's':
                print("Seite übersprungen.")
                annotations = None
                break 
            elif user_input == 'b':
                if current_col_idx > 0:
                    current_col_idx -= 1
                    continue
                else:
                    print("Du bist bereits bei der ersten Kategorie.")
                    continue
            
            # **NEU: Validiere und verarbeite die numerische Eingabe**
            try:
                # Shortcut: Leere Eingabe (Enter) wird zu 0
                if user_input == '':
                    value = 0
                else:
                    value = int(user_input)

                # Prüfe, ob die Zahl im gültigen Bereich liegt
                if not (0 <= value <= 99):
                    print(f"FEHLER: Bitte eine Zahl zwischen 0 und 99 eingeben.")
                    continue # Frage erneut für dieselbe Kategorie

                # Speichere den validierten Wert
                annotations[col] = value
                current_col_idx += 1 # Gehe zur nächsten Kategorie

            except ValueError:
                # Fängt Fehler ab, wenn die Eingabe keine Zahl ist (z.B. "abc")
                print("FEHLER: Ungültige Eingabe. Bitte eine ganze Zahl eingeben.")
                continue # Frage erneut für dieselbe Kategorie

        if annotations is not None:
            for col, value in annotations.items():
                df.loc[index, col] = value
            
            df.to_csv(csv_file, index=False)
            print(f"Annotation für Seite {index + 1} gespeichert!")
            
    print("\nAlle Seiten in diesem Subset wurden bearbeitet.")

# --- Hauptskript ausführen ---
if __name__ == "__main__":
    if not os.path.exists(CSV_FILE_TO_ANNOTATE):
        print(f"FEHLER: Die Datei '{CSV_FILE_TO_ANNOTATE}' wurde nicht gefunden.")
    else:
        dataframe = prepare_csv_for_annotation(CSV_FILE_TO_ANNOTATE)
        start_annotation_session(dataframe, CSV_FILE_TO_ANNOTATE)