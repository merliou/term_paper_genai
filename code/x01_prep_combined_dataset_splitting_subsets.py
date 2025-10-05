#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 10:56:13 2025

@author: merlin
"""
import os
import pandas as pd
import fitz  # PyMuPDF
import numpy as np

def create_dataset_from_specific_structure(root_folder, output_csv_path):
    """
    Erstellt ein initiales Dataset aus einer spezifischen Ordnerstruktur von PDFs.

    Args:
        root_folder (str): Der Pfad zum Hauptordner, der die PDFs enthält.
        output_csv_path (str): Der Pfad zum Speichern der resultierenden CSV-Datei.

    Returns:
        pd.DataFrame: Ein DataFrame mit den Metadaten der gefundenen PDFs.
    """
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
                                date_stamp = parts[-1]
                                
                                if len(date_stamp) == 4 and date_stamp.isdigit():
                                    day = date_stamp[:2]
                                    month = date_stamp[2:]
                                    year = '2025' # Festes Jahr
                                    
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
                                    print(f"Warnung: Datumsstempel im Dateinamen '{filename}' hat nicht das Format 'TTMM'.")
                            else:
                                print(f"Warnung: Dateiname '{filename}' entspricht nicht dem Schema 'name_land_TTMM.pdf'.")

    if not metadata_list:
        print("Warnung: Keine PDFs gefunden, die dem Schema entsprechen.")
        return pd.DataFrame()

    df = pd.DataFrame(metadata_list)
    df.to_csv(output_csv_path, index=False)
    print(f"Initiales Dataset wurde unter {output_csv_path} gespeichert.")
    return df

def split_pdfs_into_pages(metadata_csv_path, output_folder):
    """
    Liest ein Dataset mit PDF-Pfaden, teilt jede PDF in einzelne Seiten auf
    und erstellt ein neues Dataset, das alle Metadaten enthält.

    Args:
        metadata_csv_path (str): Pfad zur CSV-Datei aus Schritt 1.
        output_folder (str): Ordner zum Speichern der einzelnen PDF-Seiten.
        
    Returns:
        pd.DataFrame: Ein DataFrame mit den Metadaten der einzelnen Seiten.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    df = pd.read_csv(metadata_csv_path)
    split_pages_data = []
    
    for index, row in df.iterrows():
        original_path = row['original_pdf_path']
        
        try:
            doc = fitz.open(original_path)
            for page_num in range(len(doc)):
                new_doc = fitz.open()
                new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
                
                base_filename = os.path.splitext(os.path.basename(original_path))[0]
                output_filename = f"{base_filename}_page_{page_num + 1}.pdf"
                output_path = os.path.join(output_folder, output_filename)
                
                new_doc.save(output_path)
                new_doc.close()
                
                page_metadata = row.to_dict()
                page_metadata['page_number'] = page_num + 1
                page_metadata['page_pdf_path'] = output_path
                split_pages_data.append(page_metadata)
            doc.close()
        except Exception as e:
            print(f"Fehler bei der Verarbeitung von {original_path}: {e}")
            
    split_df = pd.DataFrame(split_pages_data)
    output_csv_path = 'split_pages_dataset.csv'
    split_df.to_csv(output_csv_path, index=False)
    print(f"Dataset der Einzelseiten wurde unter {output_csv_path} gespeichert.")
    return split_df

def create_shuffled_subsets(full_dataset_path, subset_size, subsets_output_folder):
    """
    Teilt das gesamte Dataset in zufällig gemischte Subsets auf, um alle Seiten zu annotieren.

    Args:
        full_dataset_path (str): Pfad zum CSV-Gesamtdatensatz der gesplitteten Seiten.
        subset_size (int): Die Anzahl der Seiten in jedem Subset.
        subsets_output_folder (str): Der Ordner zum Speichern der Subset-CSVs.
    """
    if not os.path.exists(subsets_output_folder):
        os.makedirs(subsets_output_folder)
        
    df = pd.read_csv(full_dataset_path)
    
    # Mische das gesamte DataFrame zufällig
    df_shuffled = df.sample(frac=1, random_state=np.random.RandomState()).reset_index(drop=True)
    
    # Teile das gemischte DataFrame in gleich große Chunks (Subsets)
    num_subsets = int(np.ceil(len(df_shuffled) / subset_size))
    
    for i in range(num_subsets):
        start_index = i * subset_size
        end_index = start_index + subset_size
        subset = df_shuffled.iloc[start_index:end_index]
        
        subset_filename = f"subset_{i+1}.csv"
        subset_path = os.path.join(subsets_output_folder, subset_filename)
        subset.to_csv(subset_path, index=False)
        print(f"Subset {i+1} mit {len(subset)} Seiten wurde unter {subset_path} gespeichert.")

# --- Hauptskript ---
if __name__ == "__main__":
    # 1. Erstelle das initiale Dataset aus der Ordnerstruktur
    ROOT_BROCHURES_FOLDER = 'data_term_paper/prospekte_v02'
    OUTPUT_METADATA_CSV = 'initial_dataset_new_v02.csv'
    initial_df = create_dataset_from_specific_structure(ROOT_BROCHURES_FOLDER, OUTPUT_METADATA_CSV)
    
    if not initial_df.empty:
        print("\nDie ersten Zeilen des initialen Datasets:")
        print(initial_df.head())
        
        # 2. Teile die PDFs in einzelne Seiten auf
        SPLIT_PAGES_FOLDER = 'split_pages_v02'
        split_df = split_pdfs_into_pages(OUTPUT_METADATA_CSV, SPLIT_PAGES_FOLDER)
        print("\nDie ersten Zeilen des Datasets der Einzelseiten:")
        print(split_df.head())
        
        # 3. Teile alle Seiten in gemischte Subsets auf
        SUBSETS_OUTPUT_FOLDER = 'subsets_for_annotation'
        SUBSET_SIZE = 50 # Jedes Subset hat 50 Seiten
        create_shuffled_subsets('split_pages_dataset.csv', SUBSET_SIZE, SUBSETS_OUTPUT_FOLDER)