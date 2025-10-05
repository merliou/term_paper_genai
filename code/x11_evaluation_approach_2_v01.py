import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score
import warnings

def evaluate_predictions(file_path):
    """
    Lädt eine CSV-Datei, vergleicht Vorhersagespalten mit Goldstandard-Spalten
    und berechnet Cohen's Kappa sowie den gewichteten F1-Score.

    Annahmen:
    - Die zu vergleichenden Spalten folgen dem Muster 'var' und 'var_gold'.
    - Werte wie 98.0 und 99.0 in den Goldstandard-Spalten markieren Zeilen,
      die von der Auswertung ausgeschlossen werden sollen.

    Args:
        file_path (str): Der Pfad zur CSV-Datei.
    """
    try:
        # Laden der CSV-Datei in einen pandas DataFrame
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Fehler: Die Datei unter '{file_path}' wurde nicht gefunden.")
        return

    # Deaktivieren von Warnungen für Metriken bei Klassen ohne Vorhersagen
    warnings.filterwarnings('ignore', category=UserWarning)

    # Identifizieren der Basis-Spaltennamen für die Auswertung
    # (z.B. 'alc', 'product', etc.)
    base_variables = [col for col in df.columns if f"{col}_gold" in df.columns]

    print("--- Start der systematischen Evaluation ---")
    
    # Iteration über jede zu evaluierende Variable
    for var in base_variables:
        gold_col = f"{var}_gold"
        pred_col = var

        # Erstellen eines temporären DataFrames zur sicheren Bearbeitung
        temp_df = df[[gold_col, pred_col]].copy()

        # Ausschluss von Zeilen mit speziellen Werten (98.0, 99.0) im Goldstandard
        # Diese Werte deuten oft auf "nicht anwendbar" oder "ignoriere" hin
        original_count = len(temp_df)
        filtered_df = temp_df[~temp_df[gold_col].isin([98.0, 99.0])]
        filtered_count = len(filtered_df)
        
        # Extrahieren der wahren Werte (y_true) und der Vorhersagewerte (y_pred)
        y_true = filtered_df[gold_col]
        y_pred = filtered_df[pred_col]

        if y_true.empty or y_pred.empty:
            print(f"\nVariable: '{var}'")
            print("  -> Keine gültigen Daten für die Auswertung nach dem Filtern.")
            continue

        # Berechnung von Cohen's Kappa
        # Misst die Übereinstimmung unter Berücksichtigung des Zufalls
        kappa = cohen_kappa_score(y_true, y_pred)

        # Berechnung des gewichteten F1-Scores
        # 'weighted' berücksichtigt Klassenungleichgewichte
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Ausgabe der Ergebnisse
        print(f"\nVariable: '{var}'")
        print(f"  Anzahl der ausgewerteten Zeilen: {filtered_count} (von {original_count})")
        print(f"  Cohen's Kappa: {kappa:.4f}")
        print(f"  Weighted F1-Score: {f1:.4f}")

    print("\n--- Evaluation abgeschlossen ---")

# --- Hauptteil des Skripts ---
if __name__ == "__main__":
    # Bitte ersetzen Sie 'subset_1_annotated_llama3.2:11b.csv' mit dem Pfad zu Ihrer Datei,
    # falls dieser abweicht.
    csv_file_path = 'annotations_colab/subset_1_annotated_llama3.2:11b.csv'
    evaluate_predictions(csv_file_path)