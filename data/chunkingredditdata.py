from langchain.text_splitter import RecursiveCharacterTextSplitter
import json

# Load your Reddit JSON
with open('/Users/anvesha/Desktop/uiuc/projects/llmassistant/venv/data/uiuccscoursesredditscrapeddata.json', 'r') as f:
    reddit_data = json.load(f)

# Set up text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30,
    length_function=len
)

chunked_reddit = []

for post in reddit_data:
    title = post.get("title", "").strip()
    body = post.get("body", "").strip()

    if not title and not body:
        continue  # Skip empty posts

    combined_text = f"Title: {title}\nBody: {body}"

    # Split the combined text into chunks
    chunks = splitter.split_text(combined_text)

    # Store each chunk with metadata
    for i, chunk in enumerate(chunks):
        chunked_reddit.append({
            "text": chunk,
            "metadata": {
                "original_id": post.get("id", ""),
                "title": title,
                "author": post.get("authorName", ""),
                "flair": post.get("flair", ""),
                "community": post.get("communityName", ""),
                "chunk_index": i
            }
        })

# Save the chunked data
with open('/Users/anvesha/Desktop/uiuc/projects/llmassistant/venv/data/redditchunks.json', 'w') as f:
    json.dump(chunked_reddit, f, indent=2)