# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: mono
#     language: python
#     name: python3
# ---

# %%
from docling.document_converter import DocumentConverter

source = "https://www.sciencedirect.com/science/article/pii/S2772586323000126"  # document per local path or URL
source = "Dynamic Parameter Documentation - Fleet Optimization - Confluence.pdf"  # document per local path or URL
converter = DocumentConverter()
result = converter.convert(source)
print(result.document.export())  # output: "## Docling Technical Report[...]"

# %%
print(result.document.export_to_markdown())
