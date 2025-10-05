import pandas as pd
import PyPDF2
import os

def ermittle_pdf_seitenzahl(pdf_pfad):
    """
    Ermittelt die Anzahl der Seiten in einer PDF-Datei.

    Args:
        pdf_pfad (str): Der Pfad zur PDF-Datei.

    Returns:
        int: Die Anzahl der Seiten oder 0, wenn die Datei nicht gefunden
             wurde oder beschädigt ist.
    """
    if not os.path.exists(pdf_pfad):
        print(f"Warnung: Datei unter '{pdf_pfad}' nicht gefunden.")
        return 0
    try:
        with open(pdf_pfad, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            return len(reader.pages)
    except Exception as e:
        print(f"Fehler beim Verarbeiten der Datei '{pdf_pfad}': {e}")
        return 0

# Laden des Datensatzes aus der CSV-Datei
try:
    df = pd.read_csv('initial_dataset_new_v02.csv')

    # Ermitteln der Seitenanzahl für jede PDF-Datei und Speichern in einer neuen Spalte
    df['seitenanzahl'] = df['original_pdf_path'].apply(ermittle_pdf_seitenzahl)

    # Anzeigen des aktualisierten DataFrames
    print("Aktualisierter DataFrame mit Seitenanzahl:")
    print(df.to_string())

    # Optional: Speichern des aktualisierten DataFrames in einer neuen CSV-Datei
    df.to_csv('initial_dataset_new_v03.csv', index=False)
    print("\nDer aktualisierte DataFrame wurde in 'dataset_mit_seitenzahlen.csv' gespeichert.")

except FileNotFoundError:
    print("Fehler: 'initial_dataset_new_v02.csv' nicht gefunden. Stellen Sie sicher, dass sich die Datei im selben Verzeichnis wie das Skript befindet.")
except Exception as e:
    print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")