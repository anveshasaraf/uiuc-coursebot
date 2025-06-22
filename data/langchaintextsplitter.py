from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

file_path = '/Users/anvesha/Desktop/uiuc/projects/llmassistant/venv/data/courserequirements.txt'

with open(file_path, 'r') as file:
    text = file.read()

recursive_character_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30,
    length_function=len
)

# Use split_text to get raw strings
chunks = recursive_character_splitter.split_text(text)

# Add metadata
chunks_data = [
    {
        "text": chunk,
        "metadata": {
            "source": "courserequirements.txt",
            "chunk_index": i
        }
    }
    for i, chunk in enumerate(chunks)
]

# Save the chunked data
output_path = '/Users/anvesha/Desktop/uiuc/projects/llmassistant/venv/data/courserequirements_chunks.json'
with open(output_path, 'w') as f:
    json.dump(chunks_data, f, indent=2)
