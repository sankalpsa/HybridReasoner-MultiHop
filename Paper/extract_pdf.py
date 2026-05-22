import pypdf

pdf_path = r"c:\Users\USER\OneDrive\Desktop\NITK\LLS_NEW\Paper\Applied_intelligence_journal.pdf"
output_path = r"c:\Users\USER\OneDrive\Desktop\NITK\LLS_NEW\Paper\paper_text.txt"

print(f"Reading PDF from {pdf_path}...")
reader = pypdf.PdfReader(pdf_path)
print(f"Number of pages: {len(reader.pages)}")

with open(output_path, "w", encoding="utf-8") as f:
    for i, page in enumerate(reader.pages):
        print(f"Extracting page {i+1}/{len(reader.pages)}...")
        f.write(f"--- PAGE {i+1} ---\n")
        f.write(page.extract_text() or "")
        f.write("\n\n")

print(f"Successfully extracted to {output_path}!")
