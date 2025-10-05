import pandas as pd
from sklearn.metrics import cohen_kappa_score, classification_report, f1_score
import warnings

# Deaktiviert Warnungen, die bei Klassen ohne Vorhersagen auftreten können
warnings.filterwarnings('ignore', category=UserWarning)

def evaluate_model_from_csv(filepath: str, model_name: str):
    """
    Liest eine CSV-Datei ein und führt eine detaillierte Auswertung durch.
    Identifiziert automatisch 'variable' und 'variable_hum' Spaltenpaare und
    ignoriert Zeilen, bei denen für ein Paar ein Wert fehlt.
    """
    print("=" * 80)
    print(f"REPORT FÜR MODELL: {model_name} (aus Datei: {filepath})")
    print("=" * 80)

    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"FEHLER: Die Datei '{filepath}' wurde nicht gefunden.")
        return pd.DataFrame()

    # Finde alle Spalten, die als Goldstandard (_hum) dienen
    human_cols = [col for col in df.columns if col.endswith('_hum')]
    
    # Finde die zugehörigen Modell-Spalten
    model_cols = [col.replace('_hum', '') for col in human_cols]
    
    # Stelle sicher, dass die Modell-Spalten auch existieren
    model_cols_present = [col for col in model_cols if col in df.columns]
    
    results = []

    for m_col in model_cols_present:
        h_col = f"{m_col}_hum"
        
        print(f"\n--- Auswertung für Variable: '{m_col}' ---")

        # --- Datenvorverarbeitung ---
        # Wähle nur die beiden relevanten Spalten aus
        eval_df = df[[h_col, m_col]].copy()

        # Ignoriere Zeilen, bei denen entweder die menschliche Annotation oder die
        # Modell-Vorhersage für DIESES PAAR fehlt.
        eval_df.dropna(subset=[h_col, m_col], inplace=True)
        
        if eval_df.empty:
            print("Keine vollständigen Datenpaare für diese Variable vorhanden. Auswertung übersprungen.")
            continue
            
        # Konvertiere in einen einheitlichen Datentyp für den Vergleich
        # Wir verwenden hier Integer, da es sich um kategoriale Labels handelt.
        try:
            y_true = eval_df[h_col].astype(float).astype(int)
            y_pred = eval_df[m_col].astype(float).astype(int)
        except ValueError:
            print(f"Fehler bei der Konvertierung der Daten in numerische Werte für '{m_col}'. Überspringe.")
            continue

        print(f"Anzahl der ausgewerteten Zeilen (nach Filterung): {len(y_true)}")

        # --- Metriken berechnen ---
        # Cohen's Kappa
        kappa = cohen_kappa_score(y_true, y_pred)
        
        # F1-Score (Makro-Durchschnitt für fairen Vergleich)
        # zero_division=0 verhindert Fehler, falls eine Klasse nie vorhergesagt wird
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Detaillierter Klassifikationsreport
        report = classification_report(y_true, y_pred, zero_division=0)

        print(f"Cohen's Kappa (κ): {kappa:.4f}")
        print(f"Macro F1-Score:    {f1_macro:.4f}")
        print("\nDetaillierter Klassifikationsreport:")
        print(report)
        
        results.append({
            'variable': m_col,
            'cohen_kappa': kappa,
            'f1_macro': f1_macro
        })

    return pd.DataFrame(results)

# --- Skript ausführen ---
qwen_file = 'annotations_old/old_but_with_hybrid_results/subset_1_anno_v07_qwen2.5vl:3b.csv'
llava_file = 'annotations_old/old_but_with_hybrid_results/subsets_1and2combined_qwen3:4b_x_llava:7b.csv'

# Modelle evaluieren
results_qwen = evaluate_model_from_csv(qwen_file, "Qwen 2.5VL")
results_llava = evaluate_model_from_csv(llava_file, "LLaVA 7b")

# --- Ergebnisse zusammenfassen ---
if not results_qwen.empty and not results_llava.empty:
    # Führe die Ergebnisse zusammen, um einen direkten Vergleich zu ermöglichen
    summary_df = pd.merge(
        results_qwen,
        results_llava,
        on='variable',
        suffixes=('_qwen', '_llava')
    )
    
    # Umbenennen der Spalten für bessere Lesbarkeit
    summary_df.rename(columns={
        'cohen_kappa_qwen': 'κ (Qwen)',
        'f1_macro_qwen': 'F1-Macro (Qwen)',
        'cohen_kappa_llava': 'κ (LLaVA)',
        'f1_macro_llava': 'F1-Macro (LLaVA)'
    }, inplace=True)

    print("\n" + "=" * 80)
    print("VERGLEICHENDE ZUSAMMENFASSUNG DER MODELLE")
    print("=" * 80)
    # Ausgabe als String für eine saubere Formatierung
    print(summary_df[['variable', 'κ (Qwen)', 'κ (LLaVA)', 'F1-Macro (Qwen)', 'F1-Macro (LLaVA)']].to_string(index=False))
else:
    print("\nKonnte keine vergleichende Zusammenfassung erstellen, da die Ergebnisse für ein oder beide Modelle leer sind.")