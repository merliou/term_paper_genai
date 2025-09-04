import os
import pandas as pd
import fitz  # PyMuPDF
from PIL import Image

def convert_subsets_to_images_optimized(subset_folder, image_output_folder, base_dpi=150):
    """
    Konvertiert die PDF-Seiten aus jedem Subset in JPG-Bilder mit reduzierter 
    Auflösung (halb und viertel) und speichert sie in separaten Ordnern.

    Args:
        subset_folder (str): Der Ordner, der die Subset-CSV-Dateien enthält.
        image_output_folder (str): Der Hauptordner zum Speichern der Bilder.
        base_dpi (int): Die DPI-Auflösung für das qualitativ hochwertigste Bild,
                        von dem die kleineren Versionen abgeleitet werden. 150 ist
                        ein guter Startwert.
    """
    if not os.path.exists(image_output_folder):
        os.makedirs(image_output_folder)
        
    quality_levels = {
        "qualitaet_halb": 0.5,
        "qualitaet_viertel": 0.25
    }
    
    # Durch alle Subset-Dateien im Ordner iterieren
    for subset_csv in os.listdir(subset_folder):
        if not subset_csv.endswith('.csv'):
            continue
            
        subset_name = os.path.splitext(subset_csv)[0]
        subset_df = pd.read_csv(os.path.join(subset_folder, subset_csv))
        
        print(f"\n--- Verarbeite {subset_name} ---")
        
        # Erstelle die Haupt- und Unterordner für die Bilder dieses Subsets
        subset_image_folder = os.path.join(image_output_folder, subset_name)
        for level_name in quality_levels.keys():
            os.makedirs(os.path.join(subset_image_folder, level_name), exist_ok=True)
            
        # Durch jede Seite im Subset iterieren
        for index, row in subset_df.iterrows():
            pdf_path = row['page_pdf_path']
            base_filename = os.path.splitext(os.path.basename(pdf_path))[0]
            
            try:
                doc = fitz.open(pdf_path)
                page = doc.load_page(0)
                
                # 1. PDF-Seite EINMAL in eine hochauflösende Pixelmap rendern
                pix = page.get_pixmap(dpi=base_dpi)
                base_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Optional: Bilder quadratisch machen (konsistent mit altem Skript)
                # Dies kann für manche Vision-Modelle nützlich sein.
                squared_img = Image.new("RGB", (max(base_img.width, base_img.height), max(base_img.width, base_img.height)), (255, 255, 255))
                paste_x = (squared_img.width - base_img.width) // 2
                paste_y = (squared_img.height - base_img.height) // 2
                squared_img.paste(base_img, (paste_x, paste_y))
                
                # 2. Schleife durch die Qualitätsstufen und Bilder durch Skalierung erstellen
                for level_name, factor in quality_levels.items():
                    # Neue Grösse berechnen
                    new_width = int(squared_img.width * factor)
                    new_height = int(squared_img.height * factor)
                    
                    # Bild mit hoher Qualität herunterskalieren
                    resized_img = squared_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    
                    # Pfad zum Speichern des Bildes definieren
                    output_dir = os.path.join(subset_image_folder, level_name)
                    img_path = os.path.join(output_dir, f"{base_filename}.jpg")
                    
                    # Bild als JPG mit guter Qualität speichern
                    resized_img.save(img_path, 'JPEG', quality=85)

                doc.close()
            except Exception as e:
                print(f"FEHLER beim Verarbeiten von {pdf_path}: {e}")
                
        print(f"Verarbeitung von {subset_name} abgeschlossen.")
        
    print("\nAlle Subsets wurden erfolgreich in Bilder konvertiert.")


# --- Anwendung ---
# Stellen Sie sicher, dass diese Pfade relativ zu Ihrem in Spyder
# gesetzten Arbeitsverzeichnis sind.
SUBSET_FOLDER = 'subsets'
IMAGE_OUTPUT_FOLDER = 'dataset_images'

# Starten Sie den Prozess
convert_subsets_to_images_optimized(SUBSET_FOLDER, IMAGE_OUTPUT_FOLDER)