import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_markdown(markdown_content, chunk_size, chunk_overlap):
    markdown_content = re.sub(r"\[.*?\]:.*", "", markdown_content)
    separators = ["\n## ", "\n### ", "\n", " "]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=separators)

    chunks = text_splitter.split_text(markdown_content)

    # filter out any chunks containing images in base65
    filtered_chunks = [chunk for chunk in chunks if "<data" not in chunk]
    chunks = filtered_chunks

    # for i, chunk in enumerate(chunks[:5]):
    #     print(f"Chunk {i + 1}:\n{chunk}\n{'-' * 80}")
    return chunks