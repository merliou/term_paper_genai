import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import numpy as np

def analyze_annotations(file_path):
    """
    Liest ein CSV-File, vergleicht Annotationsspalten mit ihren Goldstandard-Pendants
    und gibt einen detaillierten Bericht aus.

    Args:
        file_path (str): Der Pfad zur CSV-Datei.
    """
    try:
        # Lade die Daten aus dem CSV-File
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Fehler: Die Datei unter {file_path} wurde nicht gefunden.")
        return

    # Finde die Spalten, die einen Goldstandard haben
    gold_columns = [col for col in df.columns if col.endswith('_gold')]
    variables_to_compare = [col.replace('_gold', '') for col in gold_columns]

    print("=" * 80)
    print("Umfassender Bericht zur Annotationsqualität")
    print(f"Analysierte Datei: {file_path}")
    print(f"Anzahl der Einträge: {len(df)}")
    print("=" * 80)

    # Iteriere über jede zu vergleichende Variable
    for var in variables_to_compare:
        gold_col = f"{var}_gold"
        anno_col = var

        # Stelle sicher, dass beide Spalten existieren
        if anno_col not in df.columns:
            continue

        # Extrahiere die zu vergleichenden Spaltendaten
        y_true = df[gold_col]
        y_pred = df[anno_col]

        # Ermittle alle einzigartigen Klassen (Labels) in den Daten
        labels = sorted(np.unique(np.concatenate((y_true, y_pred))))

        # Berechne die Metriken
        accuracy = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Berechne Präzision, Recall und F1-Score für jede Klasse
        p, r, f1, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)

        # Beginne mit der Ausgabe des Berichts für die aktuelle Variable
        print(f"\n--- Analyse für Variable: '{var}' ---\n")
        print(f"Gesamtgenauigkeit (Accuracy): {accuracy:.4f}")
        print("Die Genauigkeit misst den Anteil der korrekten Vorhersagen an der Gesamtzahl der Vorhersagen.\n")

        # Gib die Konfusionsmatrix aus
        print("Konfusionsmatrix:")
        header = f"{'':<12}" + " | ".join([f"Pred: {str(label):<5}" for label in labels])
        print(header)
        print("-" * len(header))
        for i, label in enumerate(labels):
            row_str = f"Gold: {str(label):<5} | "
            row_str += " | ".join([f"{count:<5}" for count in cm[i]])
            print(row_str)
        print("\nDie Zeilen entsprechen dem Goldstandard (wahre Werte), die Spalten den Annotationen (vorhergesagte Werte).\n")
        
        # Gib die detaillierten Metriken pro Klasse aus
        print("Detaillierte Metriken pro Klasse:")
        metric_header = f"{'Klasse':<10} | {'Präzision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<10}"
        print(metric_header)
        print("-" * len(metric_header))

        for i, label in enumerate(labels):
            # Support ist die Anzahl der Vorkommen der Klasse im Goldstandard
            support = s[i]
            print(f"{str(label):<10} | {p[i]:<10.4f} | {r[i]:<10.4f} | {f1[i]:<10.4f} | {support:<10}")

        print("\n" + "=" * 80)


# --- Hauptteil des Skripts ---
if __name__ == '__main__':
    # Der Dateiname, wie im Kontext angegeben.
    csv_file = 'annotations/subset_01_gemini_1.5_pro_annotated.csv'
    analyze_annotations(csv_file)