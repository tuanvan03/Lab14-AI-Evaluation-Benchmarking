import os
import glob
from src.chunking import RecursiveChunker
from src.embedding import OpenAIEmbedder
from src.store import EmbeddingStore
from src.model import Document
import json

def preprocess_text(text: str):
    """
    Split the document into:
    - title
    - story_part
    - lessons_part
    - link_document
    """
    title = "Unknown Title"
    lines = text.strip().split("\n")
    if lines and "Đọc truyện:" in lines[0]:
        title = lines[0].replace("Đọc truyện:", "").strip()
        
    story_end_idx = len(text)
    lesson_start_idx = -1
    
    lesson_idx = text.find("Bài học rút ra")
    link_idx = text.find("Xem thêm tại:")
    
    if lesson_idx != -1:
        story_end_idx = lesson_idx
        lesson_start_idx = lesson_idx + len("Bài học rút ra")
    elif link_idx != -1:
        story_end_idx = link_idx
        
    story_part = text[:story_end_idx].strip()
    
    # remove the first line if it's the title "Đọc truyện:..."
    if lines and "Đọc truyện:" in lines[0]:
        story_part = story_part[len(lines[0]):].strip()
        
    lessons_part = ""
    if lesson_start_idx != -1:
        lesson_end_idx = link_idx if link_idx != -1 else len(text)
        lessons_part = text[lesson_start_idx:lesson_end_idx].strip()

    link_document = ""
    if link_idx != -1:
        link_document = text[link_idx:].strip()
    return title, story_part, lessons_part, link_document

def main():
    documents_dir = "documents"
    if not os.path.exists(documents_dir):
        print(f"Directory '{documents_dir}' not found.")
        return

    print("Khởi tạo các module...")
    # Initialize Core Components with overlap
    chunker = RecursiveChunker(chunk_size=500, sentence_overlap=1)
    embedder = OpenAIEmbedder()
    store = EmbeddingStore(collection_name="vietnamese_tales", embedding_fn=embedder)

    all_docs = []

    # Read and parse text files
    print("Đang đọc và chunking dữ liệu...")
    txt_files = glob.glob(os.path.join(documents_dir, "*.txt"))
    for file_path in txt_files:
        filename = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        title, story_part, lessons_part, link_document = preprocess_text(text)
        
        chunk_idx = 0

        # Chunk the story
        if story_part:
            story_chunks = chunker.chunk(story_part)
            for chunk_content in story_chunks:
                doc = Document(
                    id=f"{filename}_story_{chunk_idx}",
                    content=chunk_content.strip(),
                    metadata={
                        "source": filename,
                        "title": title,
                        "type": "story",
                        "chunk_index": chunk_idx,
                        "link_document": link_document
                    }
                )
                all_docs.append(doc)
                chunk_idx += 1

        # Chunk the lessons
        if lessons_part:
            lesson_chunks = chunker.chunk(lessons_part)
            for chunk_content in lesson_chunks:
                doc = Document(
                    id=f"{filename}_lesson_{chunk_idx}",
                    content=chunk_content.strip(),
                    metadata={
                        "source": filename,
                        "title": title,
                        "type": "lesson",
                        "chunk_index": chunk_idx,
                        "link_document": link_document
                    }
                )
                all_docs.append(doc)
                chunk_idx += 1
                
        print(f" - {filename}: {chunk_idx} chunks (story + lesson)")

    print(f"Tổng số {len(all_docs)} documents đã được tạo. Đang đưa vào Chroma DB (việc này sẽ tốn thời gian gọi API của OpenAI)...")
    
    if all_docs:
        store.add_documents(all_docs)
        print(f"Đã đưa thành công vào collection '{store._collection_name}'. Hiện có total {store.get_collection_size()} docs.")
    else:
        print("Không có documents nào để lưu vào DB.")

def test_preprocessing(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    title, story_part, lessons_part, link_document = preprocess_text(text)
    print("Title:", title)
    print("Story Part:", story_part)
    print("Lessons Part:", lessons_part)
    print("Link Document:", link_document)

def test_chunking(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    title, story_part, lessons_part, link_document = preprocess_text(text)
    
    chunker = RecursiveChunker(chunk_size=1000, sentence_overlap=2)
    
    print("\n--- TEST CHUNKING FOR STORY PART ---")
    chunks = chunker.chunk(story_part)
    for i, c in enumerate(chunks):
        print(f"Chunk {i+1} (Len: {len(c)}):")
        print(c)
        print("-" * 40)

def test_get_story_chunks(filename: str):
    print("\n--- TEST GET STORY CHUNKS FROM DB ---")
    embedder = OpenAIEmbedder()
    store = EmbeddingStore(collection_name="vietnamese_tales", embedding_fn=embedder)
    
    chunks = store.get_story_chunks_by_filename(filename)
    print(f"=> Tìm thấy tổng cộng {len(chunks)} chunks cho file '{filename}'.")
    for i, chunk in enumerate(chunks):
        print(f"\n[Chunk {i}] ID: {chunk['id']}")
        # preview = chunk['content'][:150].replace('\n', ' ')
        # print(f"Content: {preview}...")
        print(json.dumps(chunk, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    # test_preprocessing("documents/caykhe.txt")
    # test_chunking("documents/caykhe.txt")
    # main()
    test_get_story_chunks("caykhe.txt")
