import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, mean_absolute_error, mean_squared_error
import warnings



# ANGEPASST: MAE und RMSE für prod_pp und prod_alc



def evaluate_predictions(file_path):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Fehler: Die Datei unter '{file_path}' wurde nicht gefunden.")
        return

    warnings.filterwarnings('ignore', category=UserWarning)

    # Identifizieren aller Variablen, die einen Goldstandard haben
    base_variables = [col for col in df.columns if f"{col}_gold" in df.columns]
    
    # Definieren, welche Variablen metrisch sind
    metric_variables = ['prod_pp', 'prod_alc']

    print("--- Start der systematischen Evaluation ---")
    
    for var in base_variables:
        gold_col = f"{var}_gold"
        pred_col = var
        
        # Temporären DataFrame erstellen und Filtern der speziellen Werte 98.0 und 99.0
        temp_df = df[[gold_col, pred_col]].copy()
        original_count = len(temp_df)
        filtered_df = temp_df[~temp_df[gold_col].isin([98.0, 99.0])]
        filtered_count = len(filtered_df)
        
        y_true = filtered_df[gold_col]
        y_pred = filtered_df[pred_col]

        if y_true.empty:
            print(f"\nVariable: '{var}'")
            print("  -> Keine gültigen Daten für die Auswertung nach dem Filtern.")
            continue

        print(f"\nVariable: '{var}'")
        print(f"  Anzahl der ausgewerteten Zeilen: {filtered_count} (von {original_count})")

        # UNTERSCHEIDUNG: Metrisch oder Kategorial?
        if var in metric_variables:
            # --- Auswertung für metrische Daten (Regression) ---
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            
            print(f"  Typ: Metrisch (Regression)")
            print(f"  MAE (Mean Absolute Error):    {mae:.4f}")
            print(f"  RMSE (Root Mean Squared Error): {rmse:.4f}")
            
        else:
            # --- Auswertung für kategoriale Daten (Klassifikation) ---
            # Für Kappa und F1 ist es sicherer, sicherzustellen, dass die Daten 
            # als Kategorien behandelt werden, auch wenn sie als 0.0/1.0 gespeichert sind.
            kappa = cohen_kappa_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            print(f"  Typ: Kategorial (Klassifikation)")
            print(f"  Cohen's Kappa:     {kappa:.4f}")
            print(f"  Weighted F1-Score: {f1:.4f}")

    print("\n--- Evaluation abgeschlossen ---")

if __name__ == "__main__":
    csv_file_path = 'annotations_colab/subsets_123_combined_annotated_llama3.2:11b.csv'
    evaluate_predictions(csv_file_path)