# -*- coding: utf-8 -*-
"""
Umfassendes Analyse-Skript zur Auswertung von Supermarktprospekten.

Dieses Skript lädt annotierte Daten aus den Subset-Dateien, bereinigt sie
und führt eine Reihe von Analysen zur Präsenz und zum Kontext von
Alkoholwerbung durch. Die Ergebnisse werden als Konsolenausgaben und
visuell ansprechende Diagramme im 'analyse_ergebnisse'-Ordner gespeichert.

**Version 3.2:** Behebt einen Absturz bei der Diagrammerstellung durch
korrekte Indexbehandlung und eine FutureWarning in Pandas.
"""

import os
import glob
import pandas as pd
import matplotlib

# Weist Matplotlib an, ein nicht-interaktives Backend zu verwenden.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# --- Globale Konfiguration ---
ANNOTATIONS_DIR = 'annotations_api_gemini_2.0_flash/'
OUTPUT_DIR = 'analyse_ergebnisse_v05/'

# --- Konfiguration für Schriftgrößen in Plots ---
BASE_FONT_SIZE = 16
TITLE_FONT_SIZE = BASE_FONT_SIZE + 4
LABEL_FONT_SIZE = BASE_FONT_SIZE + 2
TICK_FONT_SIZE = BASE_FONT_SIZE
LEGEND_FONT_SIZE = BASE_FONT_SIZE

# --- Hilfsfunktionen für Plots ---
def annotate_bars(ax, **kwargs):
    """Fügt Datenlabels über die Balken eines Seaborn-Barplots hinzu."""
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.1f}%",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 9),
                    textcoords='offset points',
                    fontsize=TICK_FONT_SIZE, # Angepasste Schriftgröße
                    **kwargs)

# --- Hauptfunktionen ---
def load_data(annotations_dir):
    """
    Lädt alle annotierten CSV-Dateien aus dem angegebenen Verzeichnis
    und führt sie zu einem einzigen DataFrame zusammen.
    """
    if not os.path.exists(annotations_dir):
        print(f"Fehler: Das Annotations-Verzeichnis '{annotations_dir}' wurde nicht gefunden.")
        return None

    csv_files = glob.glob(os.path.join(annotations_dir, '*.csv'))
    if not csv_files:
        print(f"Fehler: Keine CSV-Dateien im Verzeichnis '{annotations_dir}' gefunden.")
        return None

    df_list = [pd.read_csv(file) for file in csv_files]
    combined_df = pd.concat(df_list, ignore_index=True)

    print(f"Daten erfolgreich geladen: {len(combined_df)} Seiten aus {combined_df['original_pdf_path'].nunique()} Prospekten.")
    return combined_df


def preprocess_data(df):
    """
    Bereinigt die Daten, konvertiert Datentypen und fügt nützliche Spalten hinzu.
    """
    for col in ['prod_pp', 'prod_alc', 'child', 'reduc', 'warning', 'alc', 'product']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].replace({98: pd.NA, 99: pd.NA})

    df.dropna(subset=['prod_pp', 'prod_alc', 'alc', 'country', 'supermarket'], inplace=True)
    df[['prod_pp', 'prod_alc']] = df[['prod_pp', 'prod_alc']].astype(int)

    df['total_pages'] = df.groupby('original_pdf_path')['page_number'].transform('max')
    df['relative_page_pos'] = (df['page_number'] / df['total_pages']).round(2)
    
    df['country'] = df['country'].astype('category')
    df['supermarket'] = df['supermarket'].astype('category')

    return df


def generate_visual_report(df):
    """
    Führt alle Analysen durch und generiert hochwertige Diagramme.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    sns.set_theme(style="whitegrid", palette="viridis")
    print("\n--- Starte Erstellung des visuellen Berichts ---")

    # --- 1. Präsenz von Alkoholwerbung nach Land ---
    print("1. Analysiere Präsenz von Alkoholwerbung nach Land...")
    alc_presence = df.groupby('country', observed=False)['alc'].value_counts(normalize=True).mul(100).rename('percent').reset_index()
    alc_presence = alc_presence[alc_presence['alc'] == 1.0]
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(x='country', y='percent', data=alc_presence.sort_values('percent', ascending=False))
    ax.set_title('Anteil der Prospektseiten mit Alkoholwerbung nach Land', fontsize=TITLE_FONT_SIZE, pad=20)
    ax.set_xlabel('Land', fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel('Anteil in Prozent (%)', fontsize=LABEL_FONT_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    annotate_bars(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '1_anteil_alkohol_land.png'))
    plt.close()

    # --- 2. Platzierung von Alkoholwerbung im Prospekt ---
    print("2. Analysiere Platzierung von Alkoholwerbung...")
    df_alc = df[df['alc'] == 1.0]
    plt.figure(figsize=(12, 7))
    ax = sns.boxplot(x='country', y='relative_page_pos', data=df_alc, showmeans=True)
    ax.set_title('Verteilung von Alkoholwerbung innerhalb der Prospekte (nach Land)', fontsize=TITLE_FONT_SIZE, pad=20)
    ax.set_xlabel('Land', fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel('Relative Seitenposition (0=Anfang, 1=Ende)', fontsize=LABEL_FONT_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '2_platzierung_alkohol_land_boxplot.png'))
    plt.close()

    # --- 3. Anteil alkoholischer Produkte an Gesamtprodukten ---
    print("3. Analysiere Anteil alkoholischer Produkte...")
    prod_summary = df.groupby('country', observed=False)[['prod_pp', 'prod_alc']].sum()
    prod_summary['alc_ratio_percent'] = (prod_summary['prod_alc'] / prod_summary['prod_pp']) * 100
    prod_summary.sort_values('alc_ratio_percent', ascending=False, inplace=True)
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(x=prod_summary.index, y='alc_ratio_percent', data=prod_summary)
    ax.set_title('Anteil alkoholischer Produkte an der Gesamtproduktzahl nach Land', fontsize=TITLE_FONT_SIZE, pad=20)
    ax.set_xlabel('Land', fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel('Anteil in Prozent (%)', fontsize=LABEL_FONT_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    annotate_bars(ax)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '3_anteil_alkoholprodukte_land.png'))
    plt.close()

    # --- 4. Analyse nach Supermarkt (ALLE) ---
    print("4. Analysiere Supermärkte...")
    supermarket_summary = df.groupby('supermarket', observed=False).agg(
        total_products=('prod_pp', 'sum'),
        alc_products=('prod_alc', 'sum')
    ).reset_index()
    supermarket_summary = supermarket_summary[supermarket_summary['total_products'] > 0]
    supermarket_summary['alc_prod_ratio'] = (supermarket_summary['alc_products'] / supermarket_summary['total_products']) * 100
    all_supermarkets_sorted = supermarket_summary.sort_values('alc_prod_ratio', ascending=False)
    num_supermarkets = len(all_supermarkets_sorted)
    plt.figure(figsize=(12, max(8, num_supermarkets * 0.5))) # Höhe angepasst für größere Schrift
    ax = sns.barplot(x='alc_prod_ratio', y='supermarket', data=all_supermarkets_sorted, orient='h', palette='magma')
    ax.set_title('Anteil alkoholischer Produkte nach Supermarkt (Alle)', fontsize=TITLE_FONT_SIZE, pad=20)
    ax.set_xlabel('Anteil alkoholischer Produkte in Prozent (%)', fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel('Supermarkt', fontsize=LABEL_FONT_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '4_alle_supermaerkte_alkoholanteil.png'))
    plt.close()

    # --- 5. Analyse von Rabattaktionen (Seiten mit vs. ohne Alkohol) ---
    print("5. Analysiere Rabattaktionen...")
    reduc_analysis = df.groupby(['country', 'alc'], observed=False)['reduc'].value_counts(normalize=True).mul(100).rename('percent').reset_index()
    reduc_analysis = reduc_analysis[reduc_analysis['reduc'] == 1.0]
    reduc_analysis['alc'] = reduc_analysis['alc'].map({0.0: 'Ohne Alkohol', 1.0: 'Mit Alkohol'})
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x='country', y='percent', hue='alc', data=reduc_analysis, palette='rocket')
    ax.set_title('Anteil der Seiten mit Rabattaktionen (Vergleich)', fontsize=TITLE_FONT_SIZE, pad=20)
    ax.set_xlabel('Land', fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel('Anteil der Seiten mit Rabatten (%)', fontsize=LABEL_FONT_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
    ax.legend(title='Seitentyp', fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '5_rabattaktionen_vergleich.png'))
    plt.close()
    
    # --- 6. Analyse von Warnhinweisen pro Land ---
    print("6. Analysiere Warnhinweise...")
    warning_analysis = df_alc.groupby('country', observed=False)['warning'].value_counts(normalize=True).mul(100).rename('percent').reset_index()
    warning_analysis = warning_analysis[warning_analysis['warning'] == 1.0]
    if not warning_analysis.empty:
        plt.figure(figsize=(12, 7))
        ax = sns.barplot(x='country', y='percent', data=warning_analysis.sort_values('percent', ascending=False), palette='cubehelix')
        ax.set_title('Anteil der Alkoholwerbung mit Warnhinweis nach Land', fontsize=TITLE_FONT_SIZE, pad=20)
        ax.set_xlabel('Land', fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel('Anteil in Prozent (%)', fontsize=LABEL_FONT_SIZE)
        ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
        annotate_bars(ax)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '6_warnhinweise_land.png'))
        plt.close()
    else:
        print("   -> Keine Alkoholwerbung mit Warnhinweisen für die Visualisierung gefunden.")

    #################################################################################
    ### NEUER ABSCHNITT: Ländervergleich am Beispiel von Lidl                      ###
    #################################################################################
    print("7. Führe Ländervergleich für Lidl durch...")
    df_lidl = df[df['supermarket'] == 'lidl'].copy()

    if df_lidl.empty:
        print("   -> Keine Daten für 'Lidl' gefunden. Analyse wird übersprungen.")
    else:
        # --- 7a. Vergleich von Alkoholpräsenz (Anteil Seiten) vs. Produktanteil ---
        # Metrik 1: Anteil der Seiten mit Alkohol
        lidl_page_presence = df_lidl.groupby('country', observed=False)['alc'].value_counts(normalize=True).mul(100).rename('percent').reset_index()
        lidl_page_presence = lidl_page_presence[lidl_page_presence['alc'] == 1.0]
        lidl_page_presence['Metric'] = 'Anteil Seiten mit Alkohol'

        # Metrik 2: Anteil der Alkohol-Produkte an allen Produkten
        lidl_prod_summary = df_lidl.groupby('country', observed=False)[['prod_pp', 'prod_alc']].sum()
        lidl_prod_summary['percent'] = (lidl_prod_summary['prod_alc'] / lidl_prod_summary['prod_pp']) * 100
        lidl_prod_summary = lidl_prod_summary.reset_index()
        lidl_prod_summary['Metric'] = 'Anteil Alkohol-Produkte'
        
        # Beide Metriken für das Diagramm kombinieren
        lidl_comparison_df = pd.concat([
            lidl_page_presence[['country', 'percent', 'Metric']],
            lidl_prod_summary[['country', 'percent', 'Metric']]
        ], ignore_index=True) # <<< FIX: Index neu erstellen, um Duplikate zu vermeiden

        plt.figure(figsize=(14, 8))
        ax = sns.barplot(x='country', y='percent', hue='Metric', data=lidl_comparison_df, palette='rocket')
        ax.set_title('Lidl im Ländervergleich: Alkoholpräsenz vs. Produktanteil', fontsize=TITLE_FONT_SIZE, pad=20)
        ax.set_xlabel('Land', fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel('Anteil in Prozent (%)', fontsize=LABEL_FONT_SIZE)
        ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
        ax.legend(title='Metrik', fontsize=LEGEND_FONT_SIZE, title_fontsize=LEGEND_FONT_SIZE)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, '7a_lidl_laendervergleich_praesenz_vs_anteil.png'))
        plt.close()
        
        # --- 7b. Vergleich der Platzierung von Alkoholwerbung ---
        df_lidl_alc = df_lidl[df_lidl['alc'] == 1.0]
        if not df_lidl_alc.empty:
            plt.figure(figsize=(12, 7))
            ax = sns.boxplot(x='country', y='relative_page_pos', data=df_lidl_alc, showmeans=True, palette='plasma')
            ax.set_title('Lidl im Ländervergleich: Platzierung der Alkoholwerbung', fontsize=TITLE_FONT_SIZE, pad=20)
            ax.set_xlabel('Land', fontsize=LABEL_FONT_SIZE)
            ax.set_ylabel('Relative Seitenposition (0=Anfang, 1=Ende)', fontsize=LABEL_FONT_SIZE)
            ax.tick_params(axis='both', which='major', labelsize=TICK_FONT_SIZE)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, '7b_lidl_laendervergleich_platzierung.png'))
            plt.close()
        else:
            print("   -> Keine Seiten mit Alkoholwerbung bei Lidl gefunden für die Platzierungsanalyse.")


if __name__ == '__main__':
    print("Starte Analyse der Supermarktprospekte...")
    
    master_df = load_data(ANNOTATIONS_DIR)
    
    if master_df is not None:
        processed_df = preprocess_data(master_df.copy())
        
        generate_visual_report(processed_df)
        
        print("\n--- Zusätzliche textbasierte Analysen ---")
        alc_on_cover = (processed_df[processed_df['page_number'] == 1]['alc'] == 1.0).mean() * 100
        print(f"Anteil der Titelseiten (Seite 1) mit Alkoholwerbung: {alc_on_cover:.2f}%")
        
        print("\nAnalyse abgeschlossen.")
        print(f"Ergebnisse und Diagramme wurden im Ordner '{OUTPUT_DIR}' gespeichert.")