import fitz  # PyMuPDF

def load_pdf(path: str) -> str:
    """
    Opens a PDF and extracts all text from every page.
    Returns one big string of the entire document.
    """
    doc = fitz.open(path)
    full_text = ""

    for page_num, page in enumerate(doc):
        text = page.get_text()
        full_text += f"\n--- Page {page_num + 1} ---\n{text}"

    doc.close()
    return full_text


if __name__ == "__main__":
    # Quick test — we'll use a sample PDF later
    print("PDF Loader ready.")
    print("Usage: load_pdf('your_file.pdf') → returns full text string")
