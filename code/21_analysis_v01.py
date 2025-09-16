import os
import glob
import pandas as pd
import matplotlib
# NEU: Weist Matplotlib an, ein nicht-interaktives Backend zu verwenden.
# Dies muss VOR dem Import von pyplot geschehen.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# --- Konfiguration ---
# Pfade zu den Daten. Passen Sie diese bei Bedarf an.
ANNOTATIONS_DIR = 'annotations/annotations_gemini_1.5_pro/'
INITIAL_DATASET_PATH = 'initial_dataset.csv'
OUTPUT_DIR = 'analyse_ergebnisse/'


def load_and_merge_data(annotations_dir, initial_dataset_path):
    """
    Lädt alle annotierten CSV-Dateien, führt sie zusammen und reichert sie
    mit den Länderinformationen aus dem initialen Dataset an.
    """
    # Überprüfen, ob die Verzeichnisse und Dateien existieren
    if not os.path.exists(annotations_dir):
        print(f"Fehler: Das Annotations-Verzeichnis '{annotations_dir}' wurde nicht gefunden.")
        return None
    if not os.path.exists(initial_dataset_path):
        print(f"Fehler: Die Datei '{initial_dataset_path}' wurde nicht gefunden.")
        return None

    csv_files = glob.glob(os.path.join(annotations_dir, '*.csv'))
    if not csv_files:
        print(f"Fehler: Keine CSV-Dateien im Verzeichnis '{annotations_dir}' gefunden.")
        return None

    df_list = [pd.read_csv(file) for file in csv_files]
    annotations_df = pd.concat(df_list, ignore_index=True)

    initial_df = pd.read_csv(initial_dataset_path)

    # Eindeutige Länder- und Supermarktinformationen extrahieren
    country_info = initial_df[['original_pdf_path', 'country']].drop_duplicates()

    # Daten zusammenführen
    merged_df = pd.merge(annotations_df, country_info, on='original_pdf_path', how='left')

    print(f"Daten erfolgreich geladen: {len(merged_df)} Seiten aus {merged_df['original_pdf_path'].nunique()} Prospekten.")
    return merged_df


def preprocess_data(df):
    """
    Bereinigt die Daten und fügt nützliche Spalten für die Analyse hinzu.
    """
    # Fehlerhafte Daten (Wert 98) oder unklare Werte (Wert 99) ausschließen
    # Wir behalten sie für einige Analysen, aber konvertieren sie für Berechnungen in NaN
    for col in ['prod_pp', 'prod_alc', 'child']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].replace({98: pd.NA, 99: pd.NA})

    df.dropna(subset=['prod_pp', 'prod_alc'], inplace=True)

    # Datentypen für Analyse optimieren
    df[['page_number', 'prod_pp', 'prod_alc']] = df[['page_number', 'prod_pp', 'prod_alc']].astype(int)

    # Gesamtseitenzahl pro Prospekt berechnen
    page_counts = df.groupby('original_pdf_path')['page_number'].transform('max')
    df['total_pages'] = page_counts
    df['relative_page_pos'] = (df['page_number'] / df['total_pages']).round(2)

    return df


def generate_report(df):
    """
    Führt alle Analysen durch und generiert Tabellen und Diagramme.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Setzt einen ansprechenden Stil für die Diagramme
    sns.set_theme(style="whitegrid")

    # --- 1. Wie präsent sind Alkoholprodukte? ---
    print("\n--- Analyse 1: Präsenz von Alkoholwerbung ---")
    alc_pages_by_country = df.groupby('country')['alc'].value_counts(normalize=True).unstack().fillna(0)
    alc_pages_by_country['alc_presence_percent'] = alc_pages_by_country[1.0] * 100
    print("Prozentualer Anteil der Seiten mit Alkoholwerbung pro Land:")
    print(alc_pages_by_country[['alc_presence_percent']].sort_values(by='alc_presence_percent', ascending=False))

    plt.figure(figsize=(10, 6))
    sns.barplot(x=alc_pages_by_country.index, y='alc_presence_percent', data=alc_pages_by_country, palette="viridis")
    plt.title('Anteil der Prospektseiten mit Alkoholwerbung nach Land', fontsize=16)
    plt.ylabel('Anteil in Prozent (%)')
    plt.xlabel('Land')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '1_anteil_seiten_mit_alkohol_land.png'))
    plt.close()

    # --- 2. Wo werden Alkoholprodukte platziert? ---
    print("\n--- Analyse 2: Platzierung von Alkoholwerbung ---")
    df_alc = df[df['alc'] == 1.0]
    print(f"Anteil der Titelseiten (Seite 1) mit Alkohol: { (df_alc['page_number'] == 1).mean() * 100:.2f}%")

    plt.figure(figsize=(12, 7))
    sns.histplot(data=df_alc, x='relative_page_pos', hue='country', multiple='stack', bins=20, palette='plasma')
    plt.title('Verteilung von Alkoholwerbung innerhalb der Prospekte', fontsize=16)
    plt.xlabel('Relative Seitenposition (0 = Anfang, 1 = Ende)')
    plt.ylabel('Anzahl der Seiten')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '2_platzierung_alkoholwerbung_prospekt.png'))
    plt.close()

    # --- 3. Anteil alkoholischer Produkte an Gesamtprodukten ---
    print("\n--- Analyse 3: Anteil alkoholischer Produkte ---")
    # Ignoriere Seiten ohne Produkte für eine faire Berechnung
    prod_analysis_df = df[df['prod_pp'] > 0]
    country_prod_summary = prod_analysis_df.groupby('country')[['prod_pp', 'prod_alc']].sum()
    country_prod_summary['alc_ratio_percent'] = (country_prod_summary['prod_alc'] / country_prod_summary['prod_pp']) * 100
    print("Anteil alkoholischer Produkte an allen Produkten pro Land:")
    print(country_prod_summary.sort_values('alc_ratio_percent', ascending=False))

    plt.figure(figsize=(10, 6))
    sns.barplot(x=country_prod_summary.index, y='alc_ratio_percent', data=country_prod_summary, palette="magma")
    plt.title('Anteil alkoholischer Produkte an der Gesamtproduktzahl nach Land', fontsize=16)
    plt.ylabel('Anteil in Prozent (%)')
    plt.xlabel('Land')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '3_anteil_alkoholprodukte_land.png'))
    plt.close()

    # --- 4. Der "alkoholischste" Supermarkt ---
    print("\n--- Analyse 4: Der 'alkoholischste' Supermarkt ---")
    supermarket_summary = df.groupby('supermarket').agg(
        total_pages=('page_number', 'count'),
        alc_pages=('alc', lambda x: x.sum()),
        total_products=('prod_pp', 'sum'),
        alc_products=('prod_alc', 'sum')
    )
    supermarket_summary['alc_page_ratio'] = (supermarket_summary['alc_pages'] / supermarket_summary['total_pages']) * 100
    supermarket_summary['alc_prod_ratio'] = (supermarket_summary['alc_products'] / supermarket_summary['total_products']) * 100
    print("Analyse nach Supermarkt (Anteil Seiten und Produkte mit Alkohol):")
    print(supermarket_summary[['alc_page_ratio', 'alc_prod_ratio']].sort_values('alc_prod_ratio', ascending=False))

    # --- 5. Nähe zu Kinderprodukten ---
    print("\n--- Analyse 5: Nähe zu Kinderprodukten ---")
    df_sorted = df.sort_values(by=['original_pdf_path', 'page_number']).reset_index(drop=True)
    df_sorted['child_on_prev_page'] = df_sorted.groupby('original_pdf_path')['child'].shift(1)
    df_sorted['child_on_next_page'] = df_sorted.groupby('original_pdf_path')['child'].shift(-1)

    df_alc_context = df_sorted[df_sorted['alc'] == 1.0].copy()
    df_alc_context['child_on_same_page'] = df_alc_context['child'] == 1.0
    df_alc_context['child_on_adjacent_page'] = (df_alc_context['child_on_prev_page'] == 1.0) | (df_alc_context['child_on_next_page'] == 1.0)
    
    total_alc_pages = len(df_alc_context)
    if total_alc_pages > 0:
        same_page_count = df_alc_context['child_on_same_page'].sum()
        adjacent_page_count = df_alc_context['child_on_adjacent_page'].sum()
        
        print(f"Auf {same_page_count / total_alc_pages * 100:.2f}% der Seiten mit Alkohol ist auch ein Kinderprodukt.")
        print(f"Bei {adjacent_page_count / total_alc_pages * 100:.2f}% der Seiten mit Alkohol ist auf der Vor- oder Folgeseite ein Kinderprodukt.")
    else:
        print("Keine Seiten mit Alkohol gefunden, um die Nähe zu Kinderprodukten zu analysieren.")


    # --- 6. Zusätzliche Analysen: Rabatte und Warnhinweise ---
    print("\n--- Analyse 6: Rabatte und Warnhinweise ---")
    print("Vergleich von Rabattaktionen auf Seiten mit/ohne Alkohol:")
    print(df.groupby('alc')['reduc'].value_counts(normalize=True).unstack().apply(lambda x: x.map('{:.2%}'.format)))
    
    if not df_alc.empty:
        print("\nAnteil der Alkoholwerbung MIT Warnhinweis pro Land:")
        warning_summary = df_alc.groupby('country')['warning'].value_counts(normalize=True).unstack().fillna(0)
        if 1.0 in warning_summary.columns:
            warning_summary['warning_present_percent'] = warning_summary[1.0] * 100
            print(warning_summary[['warning_present_percent']].sort_values('warning_present_percent', ascending=False))
        else:
            print("Keine Seiten mit Alkoholwerbung hatten einen Warnhinweis.")
    else:
        print("\nKeine Daten für Alkoholwerbung vorhanden, um Warnhinweise zu analysieren.")


if __name__ == '__main__':
    # Hauptprogramm ausführen
    print("Starte Analyse der Supermarktprospekte...")
    
    # 1. Daten laden und zusammenführen
    master_df = load_and_merge_data(ANNOTATIONS_DIR, INITIAL_DATASET_PATH)
    
    if master_df is not None:
        # 2. Daten vorverarbeiten
        processed_df = preprocess_data(master_df)
        
        # 3. Bericht und Visualisierungen erstellen
        generate_report(processed_df)
        
        print(f"\nAnalyse abgeschlossen. Ergebnisse und Diagramme wurden im Ordner '{OUTPUT_DIR}' gespeichert.")