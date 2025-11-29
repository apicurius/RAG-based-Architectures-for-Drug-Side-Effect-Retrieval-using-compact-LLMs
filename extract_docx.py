#!/usr/bin/env python3
"""Extract text from a .docx file without python-docx"""
import zipfile
import xml.etree.ElementTree as ET
import sys

def extract_text_from_docx(docx_path):
    """Extract all text from a .docx file"""
    text_content = []

    try:
        # .docx files are ZIP archives
        with zipfile.ZipFile(docx_path, 'r') as docx:
            # The main document content is in word/document.xml
            xml_content = docx.read('word/document.xml')

            # Parse the XML
            root = ET.fromstring(xml_content)

            # Define namespace
            namespace = {
                'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
            }

            # Extract all text elements
            for paragraph in root.findall('.//w:p', namespace):
                para_text = []
                for text in paragraph.findall('.//w:t', namespace):
                    if text.text:
                        para_text.append(text.text)
                if para_text:
                    text_content.append(''.join(para_text))

            return '\n'.join(text_content)
    except Exception as e:
        return f"Error extracting text: {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 extract_docx.py <path_to_docx>")
        sys.exit(1)

    docx_path = sys.argv[1]
    text = extract_text_from_docx(docx_path)
    print(text)
