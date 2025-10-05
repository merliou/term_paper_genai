import pandas as pd
import numpy as np
import glob
import os
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error
)

# ==============================================================================
# --- KONFIGURATION ---
# ==============================================================================

ANNOTATION_FOLDER = 'annotations_api_gemini_2.0_flash'
GOLD_STANDARD_FILES = [
    'subset_1_annotated.csv',
    'subset_2_annotated.csv',
    'subset_3_annotated.csv'
]
CATEGORICAL_VARS = ['alc', 'product', 'warning', 'reduc', 'child']
NUMERICAL_VARS = ['prod_pp', 'prod_alc']
VALUES_TO_EXCLUDE = [98, 99]
LATEX_OUTPUT_FILE = 'evaluation_results.tex'

# ==============================================================================
# --- HILFSFUNKTIONEN (unverändert) ---
# ==============================================================================

def load_and_combine_data(folder, filenames):
    all_dfs = []
    print(f"Lade Gold-Standard-Dateien aus dem Ordner '{folder}'...")
    for filename in filenames:
        file_path = os.path.join(folder, filename)
        try:
            df = pd.read_csv(file_path)
            all_dfs.append(df)
            print(f" -> '{filename}' erfolgreich geladen ({len(df)} Zeilen).")
        except FileNotFoundError:
            print(f" -> WARNUNG: Datei nicht gefunden: {file_path}")
    if not all_dfs: return None
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"\nAlle Dateien kombiniert. Gesamtanzahl der Einträge: {len(combined_df)}")
    return combined_df


def evaluate_categorical_variable(df, var_name):
    gold_col, pred_col = f"{var_name}_gold", var_name
    df_clean = df[[gold_col, pred_col]].dropna()
    df_clean = df_clean[~df_clean[gold_col].isin(VALUES_TO_EXCLUDE)]
    df_clean = df_clean[~df_clean[pred_col].isin(VALUES_TO_EXCLUDE)]
    y_true = df_clean[gold_col].astype(int)
    y_pred = df_clean[pred_col].astype(int)
    if len(y_true) == 0: return None
    kappa = cohen_kappa_score(y_true, y_pred)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    report_dict = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    print("\n" + "=" * 80 + f"\n--- Analyse für KATEGORISCHE Variable: '{var_name}' ---\n" + "=" * 80)
    print(f"Anzahl der verglichenen validen Datenpunkte: {len(y_true)}\n")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(f"Gewichteter F1-Score (Weighted F1-Score): {weighted_f1:.4f}\n")
    print("Detaillierter Bericht pro Klasse:")
    print(classification_report(y_true, y_pred, zero_division=0))
    return {"variable": var_name, "kappa": kappa, "weighted_f1": weighted_f1, "report_dict": report_dict}


def evaluate_numerical_variable(df, var_name):
    gold_col, pred_col = f"{var_name}_gold", var_name
    df_clean = df[[gold_col, pred_col]].dropna()
    df_clean = df_clean[~df_clean[gold_col].isin(VALUES_TO_EXCLUDE)]
    df_clean = df_clean[~df_clean[pred_col].isin(VALUES_TO_EXCLUDE)]
    y_true = df_clean[gold_col].astype(float)
    y_pred = df_clean[pred_col].astype(float)
    if len(y_true) == 0: return None
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    correlation = y_true.corr(y_pred)
    print("\n" + "=" * 80 + f"\n--- Analyse für NUMERISCHE Variable: '{var_name}' ---\n" + "=" * 80)
    print(f"Anzahl der verglichenen validen Datenpunkte: {len(y_true)}\n")
    print(f"Mittlere Absolute Abweichung (MAE): {mae:.4f}")
    print(f"Wurzel der mittleren quadratischen Abweichung (RMSE): {rmse:.4f}")
    print(f"Pearson-Korrelation: {correlation:.4f}")
    return {"variable": var_name, "mae": mae, "rmse": rmse, "correlation": correlation}


# --- KORRIGIERTE FUNKTION ---
def generate_latex_report(cat_results, num_results, output_file):
    """
    Erzeugt eine .tex-Datei mit formatierten Tabellen aus den Analyseergebnissen.
    Diese Version ist mit älteren Pandas-Versionen (< 1.0.0) kompatibel.
    """
    cat_results = [r for r in cat_results if r is not None]
    num_results = [r for r in num_results if r is not None]

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("% ============================================================\n")
        f.write("% Dieser Code wurde automatisch vom Python-Skript 'evaluation_advanced.py' generiert.\n")
        f.write("% Binden Sie diese Datei in Ihr LaTeX-Dokument mit \\input{evaluation_results.tex} ein.\n")
        f.write("% Für ein professionelleres Aussehen wird das Paket \\usepackage{booktabs} empfohlen.\n")
        f.write("% ============================================================\n\n")

        if cat_results:
            df_cat_summary = pd.DataFrame(cat_results).set_index('variable')
            df_cat_summary = df_cat_summary[['kappa', 'weighted_f1']]
            df_cat_summary.columns = ["Cohen's Kappa", "F1-Score (gewichtet)"]
            f.write("\\begin{table}[htbp]\n")
            f.write("    \\centering\n")
            f.write("    \\caption{Übereinstimmungsmetriken für kategorische Variablen}\n")
            f.write("    \\label{tab:kategorisch_summary}\n")
            # --- KORREKTUR: 'booktabs' entfernt ---
            f.write(df_cat_summary.to_latex(
                column_format="lrr",
                float_format="%.4f"
            ))
            f.write("\\end{table}\n\n")

        if num_results:
            df_num_summary = pd.DataFrame(num_results).set_index('variable')
            df_num_summary = df_num_summary[['mae', 'rmse', 'correlation']]
            df_num_summary.columns = ["MAE", "RMSE", "Pearson-Korrelation"]
            f.write("\\begin{table}[htbp]\n")
            f.write("    \\centering\n")
            f.write("    \\caption{Abweichungsmetriken für numerische Zählvariablen}\n")
            f.write("    \\label{tab:numerisch_summary}\n")
            # --- KORREKTUR: 'booktabs' entfernt ---
            f.write(df_num_summary.to_latex(
                column_format="lrrr",
                float_format="%.4f"
            ))
            f.write("\\end{table}\n\n")
        
        f.write("% ============================================================\n")
        f.write("% Detaillierte Berichte für den Anhang\n")
        f.write("% ============================================================\n\n")
        
        for result in cat_results:
            var = result['variable']
            report_df = pd.DataFrame(result['report_dict']).transpose()
            if 'support' in report_df.columns:
                 report_df['support'] = report_df['support'].astype(int)
            f.write(f"\\begin{{table}}[htbp]\n")
            f.write(f"    \\centering\n")
            f.write(f"    \\caption{{Detaillierter Klassifikationsbericht für die Variable \\texttt{{{var.replace('_', '\\_')}}}}}\n")
            f.write(f"    \\label{{tab:detail_{var}}}\n")
            # --- KORREKTUR: 'booktabs' entfernt ---
            f.write(report_df.to_latex(
                column_format="lrrrr",
                float_format="%.4f"
            ))
            f.write(f"\\end{{table}}\n\n")

    print(f"\n\nLaTeX-Bericht wurde erfolgreich in der Datei '{output_file}' gespeichert.")


# ==============================================================================
# --- HAUPTSKRIPT (unverändert) ---
# ==============================================================================
if __name__ == '__main__':
    full_df = load_and_combine_data(ANNOTATION_FOLDER, GOLD_STANDARD_FILES)
    if full_df is None:
        print("\nKeine Daten zum Analysieren gefunden.")
        exit()

    print("\n\n" + "#" * 80)
    print("### UMFASSENDE WISSENSCHAFTLICHE EVALUATION DER ANNOTATIONSQUALITÄT ###")
    print("#" * 80)

    all_categorical_results = []
    for variable in CATEGORICAL_VARS:
        res = evaluate_categorical_variable(full_df, variable)
        all_categorical_results.append(res)

    all_numerical_results = []
    for variable in NUMERICAL_VARS:
        res = evaluate_numerical_variable(full_df, variable)
        all_numerical_results.append(res)
    
    generate_latex_report(all_categorical_results, all_numerical_results, LATEX_OUTPUT_FILE)

    print("\n" + "=" * 80)
    print("Analyse abgeschlossen.")
    print("=" * 80)