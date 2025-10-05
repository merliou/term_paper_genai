import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --- KONFIGURATION ---
# Passen Sie diese Namen an, falls Ihre Spalten oder die Datei anders heißen
ANNOTATED_CSV_FILE = 'annotations/subset_2_text_classified_qwen3:4b_3gb.csv'
HUMAN_COLUMN = 'alc_hum'             # Ihre manuelle Kodierung
TEXT_FLAG_COLUMN = 'alc_keyword_flag'  # Ergebnis von Schritt 2
FINAL_ALC_COLUMN = 'alc_keyword_flag'               # Ergebnis von Schritt 3

def evaluate_text_filter(df):
    """Vergleicht die Vorauswahl (Text-Filter) mit der manuellen Kodierung."""
    print("==========================================================")
    print(" Schritt 1: Auswertung des Text-basierten Filters")
    print("==========================================================")
    print(f"Vergleiche '{HUMAN_COLUMN}' mit '{TEXT_FLAG_COLUMN}'...\n")

    # Wir brauchen eine binäre Version der manuellen Spalte (0 = kein Alkohol, 1 = Alkohol vorhanden)
    # Ihr Codebuch: 0=nein, 1=ja ohne Warn., 2=ja mit Warn.
    # Daher ist alles > 0 als "Alkohol vorhanden" zu werten.
    human_binary = (df[HUMAN_COLUMN] > 0).astype(int)
    llm_binary = df[TEXT_FLAG_COLUMN]

    # Berechne und drucke die Genauigkeit
    accuracy = accuracy_score(human_binary, llm_binary)
    print(f"-> Gesamtgenauigkeit: {accuracy:.2%}")
    print("   (Wie oft hat der Filter die Seite korrekt als 'relevant' oder 'nicht relevant' eingestuft?)\n")

    # Berechne und drucke die Konfusionsmatrix
    print("-> Konfusionsmatrix:")
    cm = confusion_matrix(human_binary, llm_binary)
    tn, fp, fn, tp = cm.ravel()
    print(f"   True Negatives (Korrekt ignoriert):    {tn}")
    print(f"   False Positives (Fälschlich markiert):   {fp}")
    print(f"   False Negatives (Fälschlich ignoriert):  {fn}")
    print(f"   True Positives (Korrekt markiert):     {tp}\n")
    print("   (Ideal wären hohe Werte bei True Negatives/Positives und niedrige bei False Negatives/Positives)\n")

    # Drucke den Klassifizierungsbericht
    print("-> Detaillierter Bericht:")
    report = classification_report(human_binary, llm_binary, target_names=['Kein Alkohol (0)', 'Alkohol (1)'])
    print(report)
    print("   - Precision: Von allen Seiten, die der Filter markiert hat, wie viele waren korrekt?")
    print("   - Recall: Von allen Seiten, die tatsächlich Alkohol hatten, wie viele hat der Filter gefunden?")
    print("   - F1-Score: Ein Mittelwert aus Precision und Recall.")


def evaluate_final_annotation(df):
    """Vergleicht die finale Bild-Annotation mit der manuellen Kodierung."""
    print("\n==========================================================")
    print(" Schritt 2: Auswertung der finalen Bild-Annotation")
    print("==========================================================")
    
    # Wir werten nur die Zeilen aus, für die eine Bild-Annotation durchgeführt wurde.
    annotated_df = df[df[FINAL_ALC_COLUMN].notna()].copy()
    
    if annotated_df.empty:
        print("Keine Seiten wurden vom multimodalen LLM annotiert. Auswertung wird übersprungen.")
        return

    print(f"Vergleiche '{HUMAN_COLUMN}' mit '{FINAL_ALC_COLUMN}' für {len(annotated_df)} Seiten...\n")

    human_final = annotated_df[HUMAN_COLUMN]
    # Das LLM gibt manchmal Strings zurück, sicherheitshalber in Integer umwandeln
    llm_final = pd.to_numeric(annotated_df[FINAL_ALC_COLUMN], errors='coerce')

    # Berechne und drucke die Genauigkeit
    accuracy = accuracy_score(human_final, llm_final)
    print(f"-> Gesamtgenauigkeit: {accuracy:.2%}")
    print("   (Wie oft stimmt die exakte Kategorie (0, 1, 2) überein?)\n")

    # Berechne und drucke die Konfusionsmatrix
    print("-> Konfusionsmatrix (Manuell vs. LLM):")
    # Definiere Labels, um alle möglichen Werte (0, 1, 2) abzudecken
    labels = sorted(list(set(human_final) | set(llm_final)))
    cm = confusion_matrix(human_final, llm_final, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"Manuell: {i}" for i in labels], columns=[f"LLM: {i}" for i in labels])
    print(cm_df)
    print("\n   (Die Diagonale von links oben nach rechts unten zeigt die korrekten Vorhersagen pro Klasse.)\n")

    # Drucke den Klassifizierungsbericht
    print("-> Detaillierter Bericht:")
    # Wandle Labels in String-Namen um
    target_names = [f"Klasse {i}" for i in labels]
    report = classification_report(human_final, llm_final, labels=labels, target_names=target_names)
    print(report)


if __name__ == "__main__":
    # Lade die annotierte CSV-Datei
    try:
        main_df = pd.read_csv(ANNOTATED_CSV_FILE)
    except FileNotFoundError:
        print(f"FEHLER: Die Datei '{ANNOTATED_CSV_FILE}' wurde nicht gefunden.")
        print("Bitte stellen Sie sicher, dass der Dateipfad korrekt ist und das Annotations-Skript gelaufen ist.")
        exit()

    # Überprüfe, ob die notwendigen Spalten existieren
    required_cols = [HUMAN_COLUMN, TEXT_FLAG_COLUMN, FINAL_ALC_COLUMN]
    if not all(col in main_df.columns for col in required_cols):
        print(f"FEHLER: Eine oder mehrere der benötigten Spalten {required_cols} wurden nicht in der CSV gefunden.")
        exit()

    # Führe die Auswertungen durch
    evaluate_text_filter(main_df)
    evaluate_final_annotation(main_df)