import pypdf
import os

pdf_path = r"c:\Users\USER\OneDrive\Desktop\NITK\LLS_NEW\Paper\Applied_intelligence_journal.pdf"
output_dir = r"c:\Users\USER\OneDrive\Desktop\NITK\LLS_NEW\Paper"

print(f"Opening PDF: {pdf_path}")
reader = pypdf.PdfReader(pdf_path)

image_count = 0
for page_idx, page in enumerate(reader.pages):
    print(f"Checking Page {page_idx + 1} for images...")
    for img_idx, image_file_object in enumerate(page.images):
        filename = f"extracted_img_p{page_idx + 1}_{img_idx + 1}_{image_file_object.name}"
        full_path = os.path.join(output_dir, filename)
        
        # Check if the filename doesn't already end with a common extension
        if not any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp']):
            full_path += '.png'
            
        print(f"Saving image to: {full_path}")
        with open(full_path, "wb") as fp:
            fp.write(image_file_object.data)
        image_count += 1

print(f"Extraction complete! Extracted {image_count} images.")
