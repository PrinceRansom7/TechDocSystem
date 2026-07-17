# Docling Document Parsing

Docling is a tool that parses documents (PDF, DOCX, PPTX, Images, HTML, etc) and exports them to a unified format (JSON or Markdown) with rich structural elements like headings, tables, and lists.

## Features

- **Format Agnostic**: Extracts from multiple document types seamlessly.
- **Table Extraction**: Can parse complex tables and output them into clean Markdown.
- **OCR Integration**: Reads text from images embedded in PDFs.

## Basic Usage

You can parse a document from your local machine or a URL.

```python
from docling.document_converter import DocumentConverter

converter = DocumentConverter()
result = converter.convert("https://arxiv.org/pdf/2408.09869")

# Export to markdown
markdown_output = result.document.export_to_markdown()
print(markdown_output)
```

## Advanced Configuration

You can configure Docling to skip certain parsing stages if you only need plain text, which speeds up the process significantly.

```python
from docling.datamodel.pipeline_options import PdfPipelineOptions

pipeline_options = PdfPipelineOptions()
pipeline_options.do_table_structure = False
pipeline_options.do_ocr = False
```
