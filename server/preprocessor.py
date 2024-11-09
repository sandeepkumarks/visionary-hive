import os
import re
import hashlib
from concurrent.futures import ProcessPoolExecutor
import logging

logging.basicConfig(level=logging.INFO)

class Preprocessor:
    def __init__(self, input_dir="documents", output_dir="documents_structured"):
        self.input_dir = input_dir
        self.output_dir = output_dir

        # Create the output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def preprocess_documents(self):
        # Read all files from the input directory in parallel
        with ProcessPoolExecutor() as executor:
            for filename in os.listdir(self.input_dir):
                file_path = os.path.join(self.input_dir, filename)
                if os.path.isfile(file_path):
                    executor.submit(self._preprocess_document, file_path)

    def _preprocess_document(self, file_path):
        """Preprocess a single document into sections based on paragraphs"""
        base_name = os.path.basename(file_path)
        structured_file_path = os.path.join(self.output_dir, f"{base_name}_structured.txt")

        # Check if the file has already been processed by comparing file hash
        if os.path.exists(structured_file_path) and self._file_hash(file_path) == self._file_hash(structured_file_path):
            return

        with open(file_path, 'r') as f:
            text = f.read()

        # Split text into paragraphs or sections (based on double newlines or headings)
        sections = re.split(r'\n\s*\n', text.strip())

        # Create a structured version of the document
        structured_content = ""
        for i, section in enumerate(sections):
            structured_content += f"Section {i+1}:\n{section.strip()}\n\n"

        with open(structured_file_path, "w") as f:
            f.write(structured_content)

    def _file_hash(self, file_path):
        """Generate a hash for a file to check if it's already processed."""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

