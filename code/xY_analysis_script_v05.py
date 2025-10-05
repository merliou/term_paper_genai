import pandas as pd
import matplotlib
# Weist Matplotlib an, ein nicht-interaktives Backend zu verwenden, um GUI-Fehler zu vermeiden.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import numpy as np

# --- Konfiguration ---
CSV_FOLDER_PATH = 'annotations_api_gemini_2.0_flash'
OUTPUT_FOLDER = 'visualisierungen'

# Code-Mappings für eine bessere Lesbarkeit in den Grafiken
PRODUCT_MAP = {
    1.0: 'Bier & Mischgetränke',
    2.0: 'Wein & Mischgetränke',
    3.0: 'Spirituosen & Mischgetränke',
    4.0: 'Andere/Mehrere Klassen'
}
COUNTRY_MAP = {
    'cz': 'Tschechien',
    'de': 'Deutschland',
    'fr': 'Frankreich',
    'pl': 'Polen',
    'est': 'Estland'
}

def load_and_prepare_data(folder_path):
    """
    Lädt alle CSV-Dateien robust, indem jede Datei einzeln verarbeitet wird,
    um Unterschiede in der Spaltenstruktur zu bewältigen.
    Wählt intelligent zwischen 'gold'- und Standardspalten und führt alles zusammen.
    """
    if not os.path.exists(folder_path):
        print(f"Fehler: Der Ordner '{folder_path}' wurde nicht gefunden.")
        return None
    
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not all_files:
        print(f"Fehler: Keine CSV-Dateien im Ordner '{folder_path}' gefunden.")
        return None

    clean_dfs = []
    # Definieren der Spalten, die wir für die Analyse benötigen
    CORE_COLS = ['country', 'supermarket', 'page_number', 'original_pdf_path']
    DATA_COLS = ['alc', 'product', 'warning', 'reduc', 'child', 'prod_pp', 'prod_alc']

    for file in all_files:
        try:
            temp_df = pd.read_csv(file)
            
            if temp_df.empty:
                continue

            clean_df = temp_df[CORE_COLS].copy()

            for col in DATA_COLS:
                gold_col = f"{col}_gold"
                if gold_col in temp_df.columns:
                    clean_df[col] = temp_df[gold_col]
                elif col in temp_df.columns:
                    clean_df[col] = temp_df[col]
                else:
                    clean_df[col] = np.nan 

            clean_dfs.append(clean_df)
        except Exception as e:
            print(f"Warnung: Konnte Datei {os.path.basename(file)} nicht verarbeiten. Fehler: {e}")

    if not clean_dfs:
        print("Fehler: Es konnten keine Daten aus den CSV-Dateien geladen werden.")
        return None

    df = pd.concat(clean_dfs, ignore_index=True)

    df.replace([98, 99], np.nan, inplace=True)
    df.dropna(subset=['alc', 'prod_pp', 'child'], inplace=True)
    
    for col in DATA_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df['country_name'] = df['country'].map(COUNTRY_MAP)
    df['product_name'] = df['product'].map(PRODUCT_MAP)

    print("\n--- Daten erfolgreich geladen und aufbereitet ---")
    print("Anzahl der analysierten Seiten pro Land:")
    print(df['country_name'].value_counts())
    print("--------------------------------------------------")

    return df

def analyze_alcohol_share_by_country(df):
    """Analyse 1: Anteil der Prospektseiten mit Alkoholwerbung pro Land."""
    print("\n--- Analyse 1: Anteil Alkoholwerbung pro Land ---")
    df_calc = df.dropna(subset=['country_name', 'alc'])
    country_share = df_calc.groupby('country_name')['alc'].value_counts(normalize=True).unstack().fillna(0)
    country_share = country_share.rename(columns={0.0: 'Ohne Alkohol', 1.0: 'Mit Alkohol'})
    
    if 'Mit Alkohol' not in country_share.columns: country_share['Mit Alkohol'] = 0
        
    country_share['Mit Alkohol'] *= 100
    print(country_share[['Mit Alkohol']].sort_values(by='Mit Alkohol', ascending=False))

    plt.figure(figsize=(12, 7))
    sns.barplot(x=country_share.index, y=country_share['Mit Alkohol'], hue=country_share.index, palette='viridis', legend=False)
    plt.title('Prozentualer Anteil der Prospektseiten mit Alkoholwerbung', fontsize=16)
    plt.ylabel('Anteil der Seiten in %', fontsize=12)
    plt.xlabel('Land', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, '1_anteil_alkohol_pro_land.png'), dpi=300)
    plt.close()

def analyze_proximity_to_child_products(df):
    """Analyse 2: Nähe von Alkohol zu Kinderprodukten."""
    print("\n--- Analyse 2: Nähe von Alkohol zu Kinderprodukten ---")
    df_sorted = df.sort_values(by=['original_pdf_path', 'page_number']).reset_index()
    
    valid_countries = df['country_name'].dropna().unique()
    if len(valid_countries) == 0:
        print("Keine gültigen Länderdaten für die Proximity-Analyse gefunden.")
        return
        
    proximity_counts = {country: {'same_page': 0, 'adjacent_page': 0, 'total_alc_pages': 0} for country in valid_countries}

    for index, row in df_sorted.iterrows():
        if row['alc'] == 1 and pd.notna(row['country_name']):
            country = row['country_name']
            proximity_counts[country]['total_alc_pages'] += 1
            if row['child'] == 1: proximity_counts[country]['same_page'] += 1
            
            is_adjacent = False
            if index > 0 and df_sorted.loc[index - 1, 'original_pdf_path'] == row['original_pdf_path'] and df_sorted.loc[index - 1, 'child'] == 1:
                is_adjacent = True
            if not is_adjacent and index < len(df_sorted) - 1 and df_sorted.loc[index + 1, 'original_pdf_path'] == row['original_pdf_path'] and df_sorted.loc[index + 1, 'child'] == 1:
                is_adjacent = True
            if is_adjacent: proximity_counts[country]['adjacent_page'] += 1

    proximity_df = pd.DataFrame(proximity_counts).T
    proximity_df['same_page_perc'] = (proximity_df['same_page'] / proximity_df['total_alc_pages'].replace(0, 1) * 100).fillna(0)
    proximity_df['adjacent_page_perc'] = (proximity_df['adjacent_page'] / proximity_df['total_alc_pages'].replace(0, 1) * 100).fillna(0)
    
    print(proximity_df[['same_page_perc', 'adjacent_page_perc']].sort_values(by='same_page_perc', ascending=False))

    proximity_df_plot = proximity_df[proximity_df['total_alc_pages'] > 0]
    if not proximity_df_plot.empty:
        # **ÄNDERUNG**: Verwendung einer konservativeren Farbpalette
        colors = ['#3b5998', '#a9a9a9'] # Ein klares Blau und ein neutrales Grau
        proximity_df_plot[['same_page_perc', 'adjacent_page_perc']].plot(
            kind='bar', stacked=True, figsize=(14, 8), color=colors,
            title='Nähe von Alkohol zu Kinderprodukten pro Land'
        )
        plt.ylabel('Anteil der Alkoholseiten in %')
        plt.xlabel('Land')
        plt.xticks(rotation=45)
        plt.legend(['Auf derselben Seite', 'Auf benachbarter Seite'])
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FOLDER, '2_naehe_zu_kinderprodukten.png'), dpi=300)
        plt.close()

def analyze_page_position_heatmap(df):
    """Analyse 3: Heatmap der Platzierung von Alkohol im Prospekt."""
    print("\n--- Analyse 3: Platzierung von Alkohol im Prospekt ---")
    df_alc = df[df['alc'] == 1].copy()
    if df_alc.empty:
        print("Keine Alkoholwerbung für die Heatmap-Analyse gefunden.")
        return
    
    page_counts = df.groupby('original_pdf_path')['page_number'].max()
    df_alc['total_pages'] = df_alc['original_pdf_path'].map(page_counts)
    df_alc = df_alc[df_alc['total_pages'] > 1]
    df_alc['normalized_page'] = (df_alc['page_number'] / df_alc['total_pages']) * 100

    df_alc['page_bin'] = pd.cut(df_alc['normalized_page'], bins=np.arange(0, 101, 10), 
                                labels=[f'{i}-{i+10}%' for i in range(0, 100, 10)], right=False)

    heatmap_data = df_alc.groupby(['country_name', 'page_bin'], observed=True).size().unstack(fill_value=0)
    heatmap_data_normalized = heatmap_data.div(heatmap_data.sum(axis=1), axis=0).replace(np.nan, 0) * 100

    if not heatmap_data_normalized.empty:
        plt.figure(figsize=(15, 8))
        sns.heatmap(heatmap_data_normalized, annot=True, fmt=".1f", cmap='YlGnBu', linewidths=.5)
        plt.title('Heatmap: Relative Platzierung von Alkoholwerbung im Prospekt', fontsize=16)
        plt.xlabel('Position im Prospekt (in % der Gesamtlänge)', fontsize=12)
        plt.ylabel('Land', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FOLDER, '3_platzierung_heatmap.png'), dpi=300)
        plt.close()

def analyze_lidl_comparison(df):
    """
    Analyse 4: Detaillierter Vergleich von Lidl in verschiedenen Ländern.
    **ÄNDERUNG**: Speichert jeden Plot als separate Bilddatei.
    """
    print("\n--- Analyse 4: Lidl im Ländervergleich ---")
    df_lidl = df[df['supermarket'] == 'lidl'].copy()
    if df_lidl.empty:
        print("Keine Daten für Lidl gefunden.")
        return
        
    # --- Plot 1: Anteil der Seiten mit Alkohol ---
    lidl_alc_share = df_lidl.groupby('country_name')['alc'].value_counts(normalize=True).unstack().fillna(0)
    lidl_alc_share['Mit Alkohol'] = lidl_alc_share.get(1.0, 0) * 100
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=lidl_alc_share.index, y=lidl_alc_share['Mit Alkohol'], hue=lidl_alc_share.index, palette='Blues', legend=False)
    plt.title('Lidl-Vergleich: Anteil der Prospektseiten mit Alkohol', fontsize=16)
    plt.ylabel('% aller Seiten')
    plt.xlabel('Land')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, '4_lidl_vergleich_anteil.png'), dpi=300)
    plt.close()
    
    # --- Plot 2: Verteilung der Alkoholarten ---
    df_lidl_alc = df_lidl[df_lidl['alc'] == 1]
    if not df_lidl_alc.empty:
        lidl_products = df_lidl_alc.groupby('country_name')['product_name'].value_counts(normalize=True).unstack().fillna(0) * 100
        
        lidl_products.plot(kind='bar', stacked=True, figsize=(12, 8), colormap='Spectral')
        plt.title('Lidl-Vergleich: Verteilung der beworbenen Alkoholarten', fontsize=16)
        plt.ylabel('% der Alkoholanzeigen')
        plt.xlabel('Land')
        plt.xticks(rotation=45)
        plt.legend(title='Produktart', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FOLDER, '4_lidl_vergleich_arten.png'), dpi=300)
        plt.close()
    
    # --- Plot 3: Anteil der Alkoholwerbung mit Rabatt ---
    if not df_lidl_alc.empty:
        lidl_reduc = df_lidl_alc.groupby('country_name')['reduc'].value_counts(normalize=True).unstack().fillna(0)
        lidl_reduc['Mit Rabatt'] = lidl_reduc.get(1.0, 0) * 100
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=lidl_reduc.index, y=lidl_reduc['Mit Rabatt'], hue=lidl_reduc.index, palette='Greens', legend=False)
        plt.title('Lidl-Vergleich: Anteil der Alkoholwerbung mit Rabatt', fontsize=16)
        plt.ylabel('% der Alkoholanzeigen')
        plt.xlabel('Land')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FOLDER, '4_lidl_vergleich_rabatte.png'), dpi=300)
        plt.close()

def analyze_product_types_by_country(df):
    """Analyse 5: Verteilung der beworbenen Alkoholarten pro Land."""
    print("\n--- Analyse 5: Alkoholarten pro Land ---")
    df_alc = df[(df['alc'] == 1) & (df['product_name'].notna())].copy()
    if df_alc.empty:
        print("Keine Daten zu Alkoholarten gefunden.")
        return

    product_distribution = df_alc.groupby('country_name')['product_name'].value_counts(normalize=True).unstack().fillna(0) * 100
    print(product_distribution)

    product_distribution.plot(
        kind='bar', stacked=True, figsize=(14, 8), colormap='tab20b',
        title='Verteilung der beworbenen Alkoholarten pro Land'
    )
    plt.ylabel('Anteil der Alkoholwerbung in %')
    plt.xlabel('Land')
    plt.xticks(rotation=45)
    plt.legend(title='Produktart', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, '5_alkoholarten_pro_land.png'), dpi=300)
    plt.close()

def main():
    """Hauptfunktion zur Ausführung der Analyse."""
    if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
        
    df = load_and_prepare_data(CSV_FOLDER_PATH)
    
    if df is not None and not df.empty:
        analyze_alcohol_share_by_country(df)
        analyze_proximity_to_child_products(df)
        analyze_page_position_heatmap(df)
        analyze_lidl_comparison(df)
        analyze_product_types_by_country(df)
        print(f"\nAnalyse abgeschlossen. Alle Grafiken wurden im Ordner '{OUTPUT_FOLDER}' gespeichert.")
    else:
        print("Daten konnten nicht geladen werden oder der DataFrame ist leer. Analyse wird abgebrochen.")

if __name__ == '__main__':
    main()