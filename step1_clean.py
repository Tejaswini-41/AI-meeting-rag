import os
import re
import json

TRANSCRIPT_DIR = "./transcripts"
OUTPUT_FILE = "./cleaned_chunks.json"

CHUNK_SIZE = 800      # characters per chunk
CHUNK_OVERLAP = 150   # overlap between chunks

def clean_line(line):
    """Clean a single line"""
    # Remove timestamp lines like: 00:01:23.456 --> 00:01:26.789
    if re.match(r'^\d{2}:\d{2}:\d{2}\.\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}\.\d{3}', line):
        return None
    # Remove WEBVTT header
    if line.strip() in ['WEBVTT', '']:
        return None
    # Remove standalone numbers (VTT cue numbers)
    if re.match(r'^\d+$', line.strip()):
        return None
    
    cleaned = line.strip()
    
    # ✅ FIX 1: Remove speaker names like "Sridhar Komalla: "
    # Matches "Firstname Lastname: " or "Name: " at any position in line
    cleaned = re.sub(r'\b[A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+)*:\s*', '', cleaned)
    
    # Collapse extra whitespace left behind
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned if cleaned else None


def chunk_text_from_lines(lines, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Build chunks from clean lines.
    ✅ FIX 2: Overlap snaps to nearest WORD boundary, not raw character cut.
    """
    chunks = []
    current_chunk = ""

    for line in lines:
        current_chunk += " " + line

        if len(current_chunk) >= chunk_size:
            chunks.append(current_chunk.strip())

            # ✅ Snap overlap to word boundary
            overlap_text = current_chunk[-overlap:]
            # Find first space in overlap to avoid cutting mid-word
            first_space = overlap_text.find(' ')
            if first_space != -1:
                overlap_text = overlap_text[first_space:].strip()

            current_chunk = overlap_text

    # Last chunk
    if len(current_chunk.strip()) > 50:
        chunks.append(current_chunk.strip())

    return chunks


def extract_topic_from_filename(filename):
    """Extract topic name from Zoom filename"""
    match = re.search(r'_([A-Z]+)_Recording', filename)
    if match:
        return match.group(1)
    return "UNKNOWN"


def process_file_streaming(filepath):
    """Read and clean file line by line"""
    clean_lines = []

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            cleaned = clean_line(line)
            if cleaned:
                clean_lines.append(cleaned)

    return clean_lines


def process_all_transcripts():
    all_chunks = []

    files = os.listdir(TRANSCRIPT_DIR)
    if not files:
        print("❌ No files found in transcripts folder!")
        return

    for filename in files:
        filepath = os.path.join(TRANSCRIPT_DIR, filename)

        if not os.path.isfile(filepath):
            continue

        print(f"\n📄 Processing: {filename}")
        print(f"   📏 File size: {os.path.getsize(filepath) / 1024:.1f} KB")

        clean_lines = process_file_streaming(filepath)
        print(f"   🧹 Clean lines extracted: {len(clean_lines)}")

        # ✅ FIX 3: Preview cleaned lines to verify speaker names are gone
        if clean_lines:
            print(f"   👀 Sample line: {clean_lines[0][:80]}")

        chunks = chunk_text_from_lines(clean_lines)
        print(f"   ✂️  Chunks created: {len(chunks)}")

        topic = extract_topic_from_filename(filename)

        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "metadata": {
                    "source_file": filename,
                    "topic": topic,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
            })

        print(f"   ✅ Done! Topic: {topic}")

        del clean_lines
        del chunks

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2)

    print(f"\n🎉 Total chunks saved: {len(all_chunks)} → {OUTPUT_FILE}")
    return all_chunks


if __name__ == "__main__":
    process_all_transcripts()