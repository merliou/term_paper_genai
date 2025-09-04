#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 31 10:56:13 2025

@author: merlin
"""
import os
import pandas as pd
import fitz  # PyMuPDF

def split_pdfs_into_pages(metadata_csv_path, output_folder):
    """
    Liest ein Dataset mit PDF-Pfaden, teilt jede PDF in einzelne Seiten auf
    und erstellt ein neues Dataset mit den Pfaden zu den einzelnen Seiten.

    Args:
        metadata_csv_path (str): Pfad zur CSV-Datei aus Schritt 1.
        output_folder (str): Ordner zum Speichern der einzelnen PDF-Seiten.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    df = pd.read_csv(metadata_csv_path)
    split_pages_data = []
    
    for index, row in df.iterrows():
        original_path = row['original_pdf_path']
        supermarket = row['supermarket']
        date = row['date']
        
        try:
            doc = fitz.open(original_path)
            for page_num in range(len(doc)):
                new_doc = fitz.open()  # Erstelle eine neue leere PDF
                new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
                
                # Definiere einen eindeutigen Dateinamen für die Seite
                base_filename = os.path.splitext(os.path.basename(original_path))[0]
                output_filename = f"{base_filename}_page_{page_num + 1}.pdf"
                output_path = os.path.join(output_folder, output_filename)
                
                new_doc.save(output_path)
                new_doc.close()
                
                page_metadata = {
                    'supermarket': supermarket,
                    'date': date,
                    'original_pdf_path': original_path,
                    'page_number': page_num + 1,
                    'page_pdf_path': output_path
                }
                split_pages_data.append(page_metadata)
            doc.close()
        except Exception as e:
            print(f"Fehler beim Verarbeiten von {original_path}: {e}")
            
    # Erstelle das neue Dataset mit den gesplitteten Seiten
    split_df = pd.DataFrame(split_pages_data)
    output_csv_path = 'split_pages_dataset.csv'
    split_df.to_csv(output_csv_path, index=False)
    print(f"Dataset der einzelnen Seiten wurde unter {output_csv_path} gespeichert.")
    return split_df
    
# --- Anwendung ---
OUTPUT_METADATA_CSV = 'initial_dataset.csv'
SPLIT_PAGES_FOLDER = 'split_pages'
split_df = split_pdfs_into_pages(OUTPUT_METADATA_CSV, SPLIT_PAGES_FOLDER)
print(split_df.head())