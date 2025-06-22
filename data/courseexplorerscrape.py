from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import re
import json


reader = PdfReader("/Users/anvesha/Desktop/uiuc/projects/llmassistant/venv/data/schedule_2025_fall_CS.pdf")
reader2 = PdfReader("/Users/anvesha/Desktop/uiuc/projects/llmassistant/venv/data/computer-science-bs-reqs-and-sample.pdf")

# Converting the PDF content to text format
text = "\n".join(page.extract_text() for page in reader.pages)
text2 = "\n".join(page.extract_text() for page in reader2.pages)

# Normalizing text
text = re.sub(r'(?<=\w)-\n(?=\w)', '', text)  # fix hyphenated line breaks
text = re.sub(r'\n+', '\n', text)
lines = text.split('\n')

# Extracting course chunks
course_start_re = re.compile(r'^CS\s\d{3}[A-Z]?\s.+?credit:\s\d+\s+hours\.', re.IGNORECASE)
course_chunks = []
current_chunk = []

for line in lines:
    if course_start_re.match(line):
        if current_chunk:
            course_chunks.append("\n".join(current_chunk).strip())
            current_chunk = []
    current_chunk.append(line)

if current_chunk:
    course_chunks.append("\n".join(current_chunk).strip())

# Preview
# for i, chunk in enumerate(course_chunks[:5]):
#     print(f"\nCourse {i+1}:\n{chunk}\n")

file = open("/Users/anvesha/Desktop/uiuc/projects/llmassistant/venv/data/cscoursefall2025.txt", "w")
for i, chunk in enumerate(course_chunks):
    file.write(f"\nCourse {i+1}:\n{chunk}\n")

file2 = open("/Users/anvesha/Desktop/uiuc/projects/llmassistant/venv/data/courserequirements.txt", "w")
file2.write(text2)


# Convert Course Explorer data into a JSON format

parsed_courses = []

for chunk in course_chunks:
    result = {}

    header_match = re.search(r'^(CS\s*\d{3}[A-Z]?)\s+(.*?)credit:\s*(\d+)\s*hours\.', chunk, re.IGNORECASE | re.MULTILINE)
    if header_match:
        result["course_code"] = header_match.group(1).strip()
        result["title"] = header_match.group(2).strip()
        result["credit_hours"] = int(header_match.group(3))
    else:
        result["course_code"] = ""
        result["title"] = ""
        result["credit_hours"] = None

    desc_match = re.search(r'credit:\s*\d+\s*hours\.\s*(.*?)(CRN Type|Prerequisite:|$)', chunk, re.DOTALL | re.IGNORECASE)
    if desc_match:
        result["description"] = desc_match.group(1).strip().replace('\n', ' ')
    else:
        result["description"] = ""

    # Extract schedule info for all CRNs
    crn_pattern = re.compile(
        r'(\d{5})\s+(\w+)\s+(\w+)\s+(\d{2}:\d{2}\s*[AP]M)\s*-\s*(\d{2}:\d{2}\s*[AP]M)\s*\n([A-Z]+)\s*(.*?)\n',
        re.IGNORECASE
    )
    schedule_entries = []
    for match in crn_pattern.finditer(chunk):
        schedule_entries.append({
            "crn": match.group(1),
            "type": match.group(2),
            "section": match.group(3),
            "start_time": match.group(4),
            "end_time": match.group(5),
            "days": match.group(6),
            "location": match.group(7).strip()
        })
    result["schedule"] = schedule_entries

    # Instructors
    instructors = re.findall(r'^\s*([A-Za-z]+,\s*[A-Z]\.?)$', chunk, re.MULTILINE)
    result["instructors"] = instructors

    # Restrictions
    restrictions = re.findall(r'(Restricted to .*?\.)', chunk, re.IGNORECASE)
    result["restrictions"] = restrictions

    # Prerequisites
    prereq_match = re.search(r'Prerequisite:(.*?)(\.|$)', chunk, re.IGNORECASE)
    if prereq_match:
        result["prerequisites"] = prereq_match.group(1).strip()
    else:
        result["prerequisites"] = ""

    parsed_courses.append(result)

# Save parsed structured data
with open("/Users/anvesha/Desktop/uiuc/projects/llmassistant/venv/data/parsed_courses.json", "w") as f:
    json.dump(parsed_courses, f, indent=2)


"""
For scraping any website

from bs4 import BeautifulSoup
import requests
url = 'http://catalog.illinois.edu/undergraduate/engineering/computer-science-bs/#samplesequencetext'
r = requests.get(url)
soup = BeautifulSoup(r.content, 'html5lib') # If this line causes an error, run 'pip install html5lib' or install html5lib
print(soup.prettify())
"""

# Convert Course Explorer JSON Data into a usable chunks

splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30,
    length_function=len
)

chunks = []
for course in parsed_courses:
    text = f"{course['course_code']} {course['title']}. {course['description']} "
    if course.get('prerequisites'):
        text += f"Prerequisites: {course['prerequisites']}. "
    if course.get('restrictions'):
        text += "Restrictions: " + " ".join(course['restrictions']) + "."

    docs = splitter.create_documents([text])
    
    for doc_index, doc in enumerate(docs):
        chunks.append({
        "text": doc.page_content,
        "metadata": {
            "course_code": course['course_code'],
            "title": course['title'],
            "chunk_index": doc_index
        }
    })


with open("/Users/anvesha/Desktop/uiuc/projects/llmassistant/venv/data/course_chunks.json", "w") as f:
    json.dump(chunks, f, indent=2)

