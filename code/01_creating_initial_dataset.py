# -*- coding: utf-8 -*-
import os
import pandas as pd

def create_dataset_from_specific_structure(root_folder, output_csv_path):
    metadata_list = []
    
    # Ebene 1: Länder (z.B. fr, de)
    for country_code in os.listdir(root_folder):
        country_path = os.path.join(root_folder, country_code)
        
        if os.path.isdir(country_path):
            # Ebene 2: Supermärkte (z.B. auchan, lidl)
            for supermarket_name in os.listdir(country_path):
                supermarket_path = os.path.join(country_path, supermarket_name)
                
                if os.path.isdir(supermarket_path):
                    # Ebene 3: PDF-Dateien
                    for filename in os.listdir(supermarket_path):
                        if filename.lower().endswith('.pdf'):
                            file_path = os.path.join(supermarket_path, filename)
                            
                            # Extrahiere Metadaten aus dem Dateinamen
                            base_name = os.path.splitext(filename)[0]
                            parts = base_name.split('_')
                            
                            # Validierung des Dateinamenschemas
                            if len(parts) >= 3:
                                # Der letzte Teil ist der Datumsstempel 'TTMM'
                                date_stamp = parts[-1]
                                
                                if len(date_stamp) == 4 and date_stamp.isdigit():
                                    day = date_stamp[:2]
                                    month = date_stamp[2:]
                                    year = '2025' # Festes Jahr wie angegeben
                                    
                                    # Erstelle einen standardisierten Datumsstring
                                    date_str = f"{year}-{month}-{day}"
                                    
                                    metadata = {
                                        'country': country_code,
                                        'supermarket': supermarket_name,
                                        'year': year,
                                        'date': date_str,
                                        'original_pdf_path': file_path
                                    }
                                    metadata_list.append(metadata)
                                else:
                                    print(f"Warnung: Datumsstempel im Dateinamen '{filename}' hat nicht das Format 'TTMM'. Datei wird übersprungen.")
                            else:
                                print(f"Warnung: Dateiname '{filename}' entspricht nicht dem erwarteten Schema 'name_land_TTMM.pdf'. Datei wird übersprungen.")

    
    # Erstelle einen DataFrame und speichere ihn als CSV
    if not metadata_list:
        print("Warnung: Keine PDF-Dateien gefunden, die dem erwarteten Schema entsprechen.")
        return pd.DataFrame()

    df = pd.DataFrame(metadata_list)
    df.to_csv(output_csv_path, index=False)
    print(f"Initiales Dataset erfolgreich erstellt und unter {output_csv_path} gespeichert.")
    return df

ROOT_BROCHURES_FOLDER = 'data_term_paper/prospekte_v02'

# 2. Geben Sie den Namen für die Ausgabedatei an
OUTPUT_METADATA_CSV = 'initial_dataset_new.csv'

# 3. Rufen Sie die Funktion mit den exakt gleichen Variablennamen auf
initial_df = create_dataset_from_specific_structure(ROOT_BROCHURES_FOLDER, OUTPUT_METADATA_CSV)

# 4. (Optional) Zeigen Sie das Ergebnis an, wenn der Vorgang erfolgreich war
if not initial_df.empty:
    print("\nDie ersten Zeilen des erstellten Datasets:")
    print(initial_df.head())
