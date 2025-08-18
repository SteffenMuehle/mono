
**Text-based Formats:**

_Tabular:_

1. **CSV (Comma-Separated Values):** A simple tabular format that separates values with commas.

_Nested:_

1. **JSON (JavaScript Object Notation):** A lightweight format that uses a syntax inspired by JavaScript. It can represent simple data structures and arrays.
    
2. **XML (eXtensible Markup Language):** A markup language that can represent complex nested data structures.
    
3. **YAML (YAML Ain't Markup Language):** A human-friendly data serialization standard for all programming languages. It can represent simple to complex nested data structures.
    
4. **TOML (Tom's Obvious, Minimal Language):** A minimal configuration file format that's easy to read because of its simple semantics. It can represent nested data structures.
    

**Binary Formats:**

_Tabular:_

1. **Parquet:** A columnar storage file format optimized for work with big data use cases.

_Nested:_

1. **BSON (Binary JSON):** Binary representation of JSON-like documents, used in databases like MongoDB.
    
2. **Protobuf (Protocol Buffers):** Binary format developed by Google that's smaller and faster than XML or JSON. It requires a schema


- parquet files, bisschen schneller als csv. spaltenbasiert. binär.
- protobuf (wie json) als format in dem moia microservices miteinander kommunizieren
	- als dateiformat für rohdaten
	- ein zentrales repo für moia, wo das dokumentiert ist: protosjson
json
bson
xml
yaml
toml
protobuf
csv