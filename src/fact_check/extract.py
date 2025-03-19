import os
import re
import json
import logging
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
from pypdf import PdfReader
from docling.document_converter import DocumentConverter
from dotenv import load_dotenv
import argparse
import sys
from pinecone_text.sparse import BM25Encoder
import textwrap
import fitz  # PyMuPDF for image extraction
import io
from PIL import Image
import base64
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Check for test mode
if "--test" not in sys.argv:
    if not GOOGLE_API_KEY or not PINECONE_API_KEY:
        raise ValueError("Missing required API keys. Please set GOOGLE_API_KEY and PINECONE_API_KEY in .env file or use --test mode")

    # Configure Google Gemini
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # Configure Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    INDEX_NAME = "med-cite-index"
else:
    # In test mode, still configure the APIs if keys are available
    if GOOGLE_API_KEY:
        genai.configure(api_key=GOOGLE_API_KEY)
    
    if PINECONE_API_KEY:
        pc = Pinecone(api_key=PINECONE_API_KEY)
    else:
        pc = None
    
    INDEX_NAME = "med-cite-index"

def create_pinecone_index_if_not_exists():
    """Create Pinecone index if it doesn't exist"""
    try:
        if INDEX_NAME not in pc.list_indexes().names():
            logger.info(f"Creating Pinecone index: {INDEX_NAME}")
            pc.create_index(
                name=INDEX_NAME,
                dimension=768,  # Gemini embedding dimension
                metric="dotproduct",  # Use dotproduct to support sparse vectors
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            logger.info(f"Successfully created index: {INDEX_NAME}")
        else:
            # Check if the existing index supports sparse vectors
            index_info = pc.describe_index(INDEX_NAME)
            logger.info(f"Index info: {index_info}")
            
            # Log index details to debug sparse vector support
            if hasattr(index_info, 'metric') and index_info.metric != 'dotproduct':
                logger.warning(f"Existing index uses {index_info.metric} metric which may not support sparse vectors. Consider recreating with dotproduct metric.")
                
        return pc.Index(INDEX_NAME)
    except Exception as e:
        logger.error(f"Error with Pinecone: {e}")
        raise

def extract_text_from_pdf(pdf_path: str, save_markdown: bool = False) -> str:
    """Extract text from a PDF file using docling"""
    try:
        # Try using docling for better PDF processing
        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        markdown_text = result.document.export_to_markdown()
        
        # Save markdown to file if requested
        if save_markdown:
            pdf_file_path = Path(pdf_path)
            markdown_file_path = pdf_file_path.with_suffix('.md')
            with open(markdown_file_path, 'w', encoding='utf-8') as f:
                f.write(markdown_text)
            logger.info(f"Saved markdown to {markdown_file_path}")
            
        return markdown_text
    except Exception as e:
        logger.info(f"Docling conversion failed: {e}. Falling back to PyPDF.")
        # Fallback to PyPDF
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

def chunk_text_by_paragraphs(text: str, max_length: int = 2500, min_length: int = 1000, overlap: int = 1) -> List[str]:
    """Split text into paragraphs, handling markdown formatting with overlap between chunks
    
    Args:
        text: The text to split into paragraphs
        max_length: Maximum length of each paragraph (default: 1500)
        min_length: Minimum preferred length for paragraphs (default: 300)
        overlap: Number of paragraphs to overlap between chunks (default: 1)
        
    Returns:
        List of paragraph strings
    """
    # Split by double newlines or multiple newlines
    paragraphs = re.split(r'\n\s*\n', text)
    
    # Process each paragraph
    processed_paragraphs = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
            
        # Skip markdown horizontal rules
        if re.match(r'^[-*_]{3,}\s*$', p):
            continue
        
        # Add the paragraph (header or regular text)
        processed_paragraphs.append(p)
        
        # If paragraph exceeds max_length, split it
        if len(p) > max_length:
            # Remove the current paragraph
            processed_paragraphs.pop()
            
            # Split long paragraphs by sentences or chunks
            sentences = re.split(r'(?<=[.!?])\s+', p)
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= max_length:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
                else:
                    if current_chunk:
                        processed_paragraphs.append(current_chunk)
                    
                    # If a single sentence is longer than max_length, split it into chunks
                    if len(sentence) > max_length:
                        for i in range(0, len(sentence), max_length):
                            chunk = sentence[i:i+max_length]
                            processed_paragraphs.append(chunk)
                        current_chunk = ""
                    else:
                        current_chunk = sentence
            
            # Add the last chunk if there is one
            if current_chunk:
                processed_paragraphs.append(current_chunk)
    
    # Additional logic to handle very short paragraphs by potentially combining them
    # if they're related (e.g., part of a list or consecutive sections on the same topic)
    
    final_paragraphs = []
    current_combined = ""
    
    for p in processed_paragraphs:
        # Don't combine headers
        if p.startswith('#'):
            if current_combined:
                final_paragraphs.append(current_combined)
                current_combined = ""
            final_paragraphs.append(p)
        elif len(p) < min_length and len(current_combined) + len(p) + 1 <= max_length:
            # Combine very short paragraphs when possible
            if current_combined:
                current_combined += "\n\n" + p
            else:
                current_combined = p
        else:
            if current_combined:
                final_paragraphs.append(current_combined)
                current_combined = ""
            final_paragraphs.append(p)
    
    # Add any remaining combined paragraph
    if current_combined:
        final_paragraphs.append(current_combined)
    
    # Create overlapping chunks if overlap > 0
    if overlap > 0 and len(final_paragraphs) > 1:
        overlapped_paragraphs = []
        
        # Create sliding window of paragraphs
        for i in range(0, len(final_paragraphs), max(1, overlap)):
            # Create a chunk with 'overlap' paragraphs (or fewer if near the end)
            chunk_size = min(overlap * 2, len(final_paragraphs) - i)
            if chunk_size <= 0:
                break
                
            # Combine paragraphs in this chunk
            chunk_text = ""
            for j in range(chunk_size):
                if j > 0:
                    chunk_text += "\n\n"
                chunk_text += final_paragraphs[i + j]
            
            overlapped_paragraphs.append(chunk_text)
        
        # If the last chunk doesn't reach the end, add a final chunk
        if overlapped_paragraphs and i + chunk_size < len(final_paragraphs):
            final_chunk = "\n\n".join(final_paragraphs[-(overlap*2):])
            overlapped_paragraphs.append(final_chunk)
            
        return overlapped_paragraphs
        
    return final_paragraphs

def save_paragraphs(paragraphs: List[str], output_path: str, format: str = "md"):
    """Save paragraphs to a file in the specified format"""
    output_path = Path(output_path)
    
    if format.lower() == "md":
        # Save as markdown
        with open(output_path.with_suffix('.md'), 'w', encoding='utf-8') as f:
            for i, paragraph in enumerate(paragraphs):
                f.write(f"## Paragraph {i+1}\n\n")
                f.write(f"{paragraph}\n\n")
                f.write("---\n\n")
        logger.info(f"Saved paragraphs as markdown to {output_path.with_suffix('.md')}")
    
    elif format.lower() == "json":
        # Save as JSON
        paragraphs_data = [{"id": i, "text": p} for i, p in enumerate(paragraphs)]
        with open(output_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
            json.dump(paragraphs_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved paragraphs as JSON to {output_path.with_suffix('.json')}")
    
    elif format.lower() == "txt":
        # Save as plain text
        with open(output_path.with_suffix('.txt'), 'w', encoding='utf-8') as f:
            for i, paragraph in enumerate(paragraphs):
                f.write(f"=== Paragraph {i+1} ===\n\n")
                f.write(f"{paragraph}\n\n")
                f.write("="*40 + "\n\n")
        logger.info(f"Saved paragraphs as text to {output_path.with_suffix('.txt')}")
    
    else:
        logger.info(f"Unsupported format: {format}")

def get_embeddings(text: str, context_paragraphs: List[str] = None) -> Dict[str, List[float]]:
    """Get dense and sparse embeddings from Gemini with optional context
    
    Args:
        text: The main text to embed
        context_paragraphs: Optional list of surrounding paragraphs for context
    
    Returns:
        Dictionary with dense and sparse embeddings
    """
    try:
        # Prepare text with context if provided, but keep original text
        if context_paragraphs:
            # Create context string to prepend
            context_string = textwrap.dedent(f"""\
            <document>
            {' '.join(context_paragraphs)}
            </document>
            Here is the chunk we want to situate within the whole document
            <chunk>
            {text}
            </chunk>
            
            """)
            
            # Prepend context to original text
            contextual_text = context_string + text
        else:
            contextual_text = text
            
        # Use the embedding API for dense embeddings
        result = genai.embed_content(
            model="models/embedding-001",
            content=contextual_text,
            task_type="retrieval_document",
            title="Medical document"
        )
        
        # Extract dense embeddings
        dense_embedding = result["embedding"]
        
        # Generate proper sparse embeddings using BM25
        try:
            # Initialize BM25 encoder with default parameters
            bm25 = BM25Encoder.default()
            
            # Encode the text as a sparse vector
            sparse_vector = bm25.encode_documents(contextual_text)
            
            # Return both dense and proper sparse embeddings
            return {
                "dense": dense_embedding,
                "sparse": sparse_vector
            }
        except Exception as e:
            logger.error(f"Error generating sparse embeddings: {e}")
            return None
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return None

def process_pdf_directory(pdf_dir: str, batch_size: int = 100, save_markdown: bool = False, 
                       save_paragraphs_format: str = None, context_window_size: int = 2, overlap: int = 1,
                       extract_images: bool = False, save_images_dir: str = None, pdf_limit: int = None,
                       specific_pdf: str = None):
    """Process all PDFs in a directory and upload to Pinecone
    
    Args:
        pdf_dir: Directory containing PDF files
        batch_size: Batch size for uploading vectors to Pinecone
        save_markdown: Whether to save markdown output
        save_paragraphs_format: Format to save extracted paragraphs (md, json, txt, html)
        context_window_size: Number of paragraphs before and after to include as context
        overlap: Number of paragraphs to overlap between chunks
        extract_images: Whether to extract and process images
        save_images_dir: Directory to save extracted images
        pdf_limit: Maximum number of PDFs to process (None for all)
        specific_pdf: Specific PDF filename to process (None for all)
    """
    pdf_dir_path = Path(pdf_dir)
    
    # Get all PDFs in the directory
    pdf_files = list(pdf_dir_path.glob("**/*.pdf"))
    
    if not pdf_files:
        logger.info(f"No PDF files found in {pdf_dir}")
        return
    
    # Filter to specific PDF if requested
    if specific_pdf:
        pdf_files = [pdf for pdf in pdf_files if pdf.name == specific_pdf]
        if not pdf_files:
            logger.error(f"Specific PDF {specific_pdf} not found in {pdf_dir}")
            return
        logger.info(f"Processing specific PDF: {specific_pdf}")
    
    # Limit the number of PDFs if requested
    elif pdf_limit is not None and pdf_limit > 0:
        pdf_files = pdf_files[:pdf_limit]
        logger.info(f"Limited to processing {pdf_limit} PDF files")
    
    logger.info(f"Found {len(pdf_files)} PDF files")
    
    # Create Pinecone index
    index = create_pinecone_index_if_not_exists()
    
    # Check if sparse vectors are supported
    try:
        index_info = pc.describe_index(INDEX_NAME)
        supports_sparse = hasattr(index_info, 'metric') and index_info.metric == 'dotproduct'
        if not supports_sparse:
            logger.warning("Index does not support sparse vectors. Only using dense vectors.")
    except Exception:
        supports_sparse = False
        logger.warning("Could not determine if index supports sparse vectors. Only using dense vectors.")
    
    # Process PDFs
    batch = []
    uploaded_vectors = 0
    
    for pdf_index, pdf_file in enumerate(tqdm(pdf_files, desc="Processing PDFs")):
        try:
            logger.info(f"Processing PDF {pdf_index+1}/{len(pdf_files)}: {pdf_file.name}")
            
            # Extract text from PDF using docling
            logger.info(f"Extracting text from {pdf_file.name}...")
            start_time = time.time()
            text = extract_text_from_pdf(str(pdf_file), save_markdown=save_markdown)
            logger.info(f"Text extraction completed in {time.time() - start_time:.2f}s")
            
            # Chunk text by paragraphs with overlap
            logger.info(f"Chunking text into paragraphs...")
            start_time = time.time()
            paragraphs = chunk_text_by_paragraphs(text, overlap=overlap)
            logger.info(f"Created {len(paragraphs)} paragraphs with {overlap} overlap in {time.time() - start_time:.2f}s")
            
            # Save paragraphs if requested
            if save_paragraphs_format:
                output_path = pdf_file.with_name(f"{pdf_file.stem}_paragraphs")
                save_paragraphs(paragraphs, output_path, format=save_paragraphs_format)
            
            # Process each paragraph
            logger.info(f"Processing {len(paragraphs)} paragraphs...")
            for i, paragraph in enumerate(paragraphs):
                if i % 10 == 0:  # Log progress every 10 paragraphs
                    logger.info(f"  Processing paragraph {i+1}/{len(paragraphs)}")
                
                if not paragraph:
                    continue
                
                # Determine if paragraph is a header
                is_header = paragraph.startswith('#')
                header_level = 0
                if is_header:
                    header_level = len(re.match(r'^#+', paragraph).group())
                
                # Create metadata with overlap information
                metadata = {
                    "document_name": pdf_file.name,
                    "document_path": str(pdf_file),
                    "paragraph_index": i,
                    "is_header": is_header,
                    "header_level": header_level,
                    "text": paragraph,
                    "overlap": overlap,
                    "has_overlap": i > 0,  # First chunk doesn't have preceding text overlap
                    "content_type": "text"  # Mark as text content
                }
                
                # Get context paragraphs (surrounding paragraphs)
                start_idx = max(0, i - context_window_size)
                end_idx = min(len(paragraphs), i + context_window_size + 1)
                context_paragraphs = paragraphs[start_idx:i] + paragraphs[i+1:end_idx]
                
                # Get embeddings with context
                embeddings = get_embeddings(paragraph, context_paragraphs)
                
                if embeddings is None:
                    logger.warning(f"Failed to get embeddings for paragraph {i} in {pdf_file.name}")
                    continue
                
                # Create vector record
                vector_id = f"{pdf_file.stem}_p{i}"
                
                # Ensure no null values in metadata (Pinecone requirement)
                for key, value in list(metadata.items()):
                    if value is None:
                        metadata[key] = ""  # Replace None with empty string
                
                vector = {
                    "id": vector_id,
                    "values": embeddings["dense"],
                    "metadata": metadata
                }
                
                # Only add sparse values if the index supports them
                if supports_sparse:
                    vector["sparse_values"] = embeddings["sparse"]
                
                # Validate vector before appending to batch
                batch.append(vector)
                
                # Upload in batches
                if len(batch) >= batch_size:
                    logger.info(f"Uploading batch of {len(batch)} vectors to Pinecone...")
                    start_time = time.time()
                    index.upsert(vectors=batch)
                    uploaded_vectors += len(batch)
                    logger.info(f"Upload completed in {time.time() - start_time:.2f}s. Total vectors uploaded: {uploaded_vectors}")
                    batch = []
            
            # Process images if requested
            if extract_images:
                # Create images directory for this PDF if saving
                pdf_images_dir = None
                if save_images_dir:
                    pdf_images_dir = os.path.join(save_images_dir, pdf_file.stem)
                    os.makedirs(pdf_images_dir, exist_ok=True)
                
                logger.info(f"Extracting images for {pdf_file.name}...")
                # Extract images
                images = extract_images_from_pdf(str(pdf_file), output_dir=pdf_images_dir)
                logger.info(f"Found {len(images)} images in {pdf_file.name}")
                
                # Process each image
                for i, img_data in enumerate(images):
                    try:
                        logger.info(f"Processing image {i+1}/{len(images)} from {pdf_file.name}: {img_data['id']}")
                        
                        # Classify image
                        logger.info(f"  Classifying image {img_data['id']}...")
                        start_time = time.time()
                        image_type = classify_research_image(img_data["image_bytes"], img_data.get("caption"))
                        img_data["image_type"] = image_type
                        logger.info(f"  Classification complete: {image_type} (in {time.time() - start_time:.2f}s)")
                        
                        # Generate description
                        logger.info(f"  Generating description for image {img_data['id']}...")
                        start_time = time.time()
                        description = get_image_description(
                            img_data["image_bytes"], 
                            caption=img_data.get("caption")
                        )
                        logger.info(f"  Description generated in {time.time() - start_time:.2f}s")
                        
                        # Add description to metadata
                        img_data["description"] = description
                        
                        # Generate multimodal embeddings
                        logger.info(f"  Generating embeddings for image {img_data['id']}...")
                        start_time = time.time()
                        embeddings = get_multimodal_embeddings(img_data["image_bytes"], description)
                        if embeddings:
                            logger.info(f"  Embeddings generated in {time.time() - start_time:.2f}s, dimensions: {len(embeddings['dense'])}")
                        else:
                            logger.error(f"  Failed to generate embeddings for image {img_data['id']}")
                            continue  # Skip this image if embeddings failed
                        
                        # Remove image bytes from metadata before storage
                        metadata = img_data.copy()
                        metadata.pop("image_bytes", None)
                        
                        # Ensure no null values in metadata (Pinecone requirement)
                        for key, value in list(metadata.items()):
                            if value is None:
                                metadata[key] = ""  # Replace None with empty string
                        
                        # Create vector record
                        vector_id = f"{img_data['id']}_image"
                        vector = {
                            "id": vector_id,
                            "values": embeddings["dense"],
                            "metadata": metadata
                        }
                        
                        # Add to batch
                        batch.append(vector)
                        logger.info(f"  Added image vector to batch (current batch size: {len(batch)})")
                        
                        # Upload in batches
                        if len(batch) >= batch_size:
                            logger.info(f"Uploading batch of {len(batch)} vectors to Pinecone...")
                            start_time = time.time()
                            index.upsert(vectors=batch)
                            uploaded_vectors += len(batch)
                            logger.info(f"Upload completed in {time.time() - start_time:.2f}s. Total vectors uploaded: {uploaded_vectors}")
                            batch = []
                    except Exception as e:
                        logger.error(f"Error processing image {img_data['id']}: {e}")
                        continue  # Continue with next image even if this one fails
        
        except Exception as e:
            logger.error(f"Error processing {pdf_file}: {e}")
    
    # Upload any remaining vectors
    if batch:
        logger.info(f"Uploading final batch of {len(batch)} vectors to Pinecone...")
        start_time = time.time()
        index.upsert(vectors=batch)
        uploaded_vectors += len(batch)
        logger.info(f"Upload completed in {time.time() - start_time:.2f}s. Total vectors uploaded: {uploaded_vectors}")

def extract_images_from_pdf(pdf_path: str, output_dir: str = None) -> List[Dict]:
    """Extract images from a PDF file with focus on research figures
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Optional directory to save extracted images
        
    Returns:
        List of dictionaries with image data and metadata
    """
    image_list = []
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Open the PDF
        doc = fitz.open(pdf_path)
        pdf_name = Path(pdf_path).stem
        
        # Iterate through pages
        for page_num, page in enumerate(doc):
            # Get text to identify captions
            page_text = page.get_text()
            caption_matches = re.finditer(r'(figure|fig\.?|table)\s+(\d+)[:\.]?(.+?)(?=\n\n|\n[A-Z]|$)', 
                                         page_text, re.IGNORECASE)
            captions = {m.group(2): m.group(3).strip() for m in caption_matches}
            
            # Method 1: Extract images using PyMuPDF's built-in method
            image_list_per_page = page.get_images(full=True)
            
            # Process each image from Method 1
            for img_index, img_info in enumerate(image_list_per_page):
                xref = img_info[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Create PIL Image for analysis
                image = Image.open(io.BytesIO(image_bytes))
                
                # Filter out small images (likely icons or decorations)
                if image.width < 150 or image.height < 150:
                    continue
                    
                # Filter out extreme aspect ratios
                aspect_ratio = image.width / image.height
                if aspect_ratio > 5 or aspect_ratio < 0.2:
                    continue
                
                # Generate unique ID
                image_id = f"{pdf_name}_p{page_num}_img{img_index}"
                
                # Try to find a caption
                caption = None
                for fig_num, caption_text in captions.items():
                    if caption is None:
                        caption = f"Figure {fig_num}: {caption_text}"
                
                # Save image if requested
                image_path = None
                if output_dir:
                    image_path = os.path.join(output_dir, f"{image_id}.{image_ext}")
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)
                
                # Create metadata
                image_metadata = {
                    "id": image_id,
                    "document_name": pdf_name,
                    "document_path": str(pdf_path),
                    "page_number": page_num + 1,
                    "image_index": img_index,
                    "format": image_ext,
                    "width": image.width,
                    "height": image.height,
                    "caption": caption,
                    "path": image_path,
                    "image_bytes": image_bytes,
                    "content_type": "image"  # Mark as image content
                }
                
                image_list.append(image_metadata)
            
            # Method 2: Extract page as image to preserve annotations and labels
            # This captures the complete rendering including text overlays and annotations
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
            png_bytes = pix.tobytes("png")
            
            # Generate unique ID for the full page image
            page_image_id = f"{pdf_name}_p{page_num}_full"
            
            # Save full page image if requested
            page_image_path = None
            if output_dir:
                page_image_path = os.path.join(output_dir, f"{page_image_id}.png")
                with open(page_image_path, "wb") as f:
                    f.write(png_bytes)
            
            # Create PIL Image for analysis
            page_image = Image.open(io.BytesIO(png_bytes))
            
            # Create metadata for the full page image
            page_image_metadata = {
                "id": page_image_id,
                "document_name": pdf_name,
                "document_path": str(pdf_path),
                "page_number": page_num + 1,
                "image_index": "full",
                "format": "png",
                "width": page_image.width,
                "height": page_image.height,
                "caption": "Full page image with all text and annotations preserved",
                "path": page_image_path,
                "image_bytes": png_bytes,
                "content_type": "image",  # Mark as image content
                "is_full_page": True  # Mark as full page image
            }
            
            image_list.append(page_image_metadata)
            
            # Method 3: Smart figure detection based on captions
            figures = detect_and_extract_figures(page, pdf_name, page_num, output_dir)
            if figures:
                image_list.extend(figures)
                logger.info(f"Detected {len(figures)} figures on page {page_num+1}")
        
        return image_list
        
    except Exception as e:
        logger.error(f"Error extracting images from {pdf_path}: {e}")
        return []

def classify_research_image(image_data: bytes, caption: str = None) -> str:
    """Classify the type of research image
    
    Args:
        image_data: Raw image bytes
        caption: Optional caption from the paper
        
    Returns:
        String classification of the image
    """
    try:
        # Encode image as base64
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Create classification prompt
        prompt = """
        Classify this research paper image into EXACTLY ONE of these categories:
        - Table: structured arrangement of data in rows and columns
        - Chart/Graph: visual representation of data (bar, line, scatter, pie charts, etc.)
        - Diagram: illustration explaining a concept, process, or system
        - Flowchart: diagram representing a process or workflow
        - Microscopy/Medical Image: biological/medical specimen images
        - Chemical Structure: molecular or chemical structure visualization
        - Equation/Formula: mathematical expressions
        - Other: anything that doesn't fit above categories
        
        Return ONLY the category name without explanation.
        """
        
        if caption:
            prompt += f"\n\nImage caption: \"{caption}\""
        
        # Submit to Gemini
        model = genai.GenerativeModel('models/gemini-1.5-flash')
        response = model.generate_content(
            contents=[
                {
                    "role": "user", 
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {
                            "mime_type": "image/jpeg",
                            "data": base64_image
                        }}
                    ]
                }
            ]
        )
        
        # Parse response 
        result = response.text.strip()
        
        # Look for one of our categories
        categories = ["Table", "Chart/Graph", "Diagram", "Flowchart", 
                     "Microscopy/Medical Image", "Chemical Structure", "Equation/Formula", "Other"]
        
        for category in categories:
            if category.lower() in result.lower():
                return category
                
        return result
    except Exception as e:
        logger.error(f"Error classifying image: {e}")
        return "Unknown"

def get_image_description(image_data: bytes, caption: str = None, prompt: str = None) -> str:
    """Generate detailed description of research visual
    
    Args:
        image_data: Raw image bytes
        caption: Optional caption from the paper
        prompt: Optional custom prompt for image description
        
    Returns:
        String description of the image
    """
    try:
        # Encode image
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Create specialized prompt if not provided
        if not prompt:
            prompt = """
            Please analyze this image from a research paper thoroughly and provide a detailed description.
            """
            
            if caption:
                prompt += f"\nThe image caption is: \"{caption}\"\n\n"
                
            prompt += """
            Focus specifically on:
            1. The key information being conveyed 
            2. ALL text content visible in the image (transcribe tables fully)
            3. Any numeric data, measurements, or statistical information
            4. Any legends, labels, or annotations
            5. The relationships between different elements in diagrams or flowcharts
            
            Be extremely thorough in extracting ALL information from this research visual.
            """
        
        # Submit to Gemini
        model = genai.GenerativeModel('models/gemini-1.5-flash')
        response = model.generate_content(
            contents=[
                {
                    "role": "user", 
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {
                            "mime_type": "image/jpeg",
                            "data": base64_image
                        }}
                    ]
                }
            ]
        )
        
        return response.text
    except Exception as e:
        logger.error(f"Error generating image description: {e}")
        return "Error: Could not generate image description"

def get_multimodal_embeddings(image_data: bytes, description: str = None) -> Dict[str, List[float]]:
    """Generate embeddings for images based on their descriptions
    
    Args:
        image_data: Raw image bytes (not used for embedding but kept for API consistency)
        description: Text description of the image
        
    Returns:
        Dictionary with dense embeddings
    """
    try:
        # Prepare text component - ensure we have some text
        if not description or len(description.strip()) < 10:
            description = "Research paper image with insufficient description"
        
        # For now, we'll use text-only embeddings based on the image description
        # This is a limitation of the current embedding API
        result = genai.embed_content(
            model="models/embedding-001",
            content=description,
            task_type="retrieval_document",
            title="Research paper image"
        )
        
        # Extract dense embedding
        dense_embedding = result["embedding"]
        
        # Return embedding
        return {
            "dense": dense_embedding
        }
    except Exception as e:
        logger.error(f"Error generating embeddings for image: {e}")
        return None

def detect_and_extract_figures(page, pdf_name, page_num, output_dir=None):
    """Attempts to detect figure regions on a page and extract them as images
    
    Args:
        page: PyMuPDF page object
        pdf_name: Name of the PDF file
        page_num: Page number
        output_dir: Directory to save extracted images
        
    Returns:
        List of dictionaries with figure data
    """
    figure_list = []
    
    try:
        # Get text blocks that might be figure captions
        blocks = page.get_text("dict")["blocks"]
        
        # Find blocks that might be figure captions
        caption_blocks = []
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    if "spans" in line:
                        text = ""
                        for span in line["spans"]:
                            text += span["text"]
                        
                        # Look for figure/table captions
                        if re.search(r'(figure|fig\.?|table)\s+\d+', text, re.IGNORECASE):
                            caption_blocks.append({
                                "text": text,
                                "bbox": block["bbox"]
                            })
        
        # Use the caption locations to estimate figure regions
        for i, caption in enumerate(caption_blocks):
            # Match the figure number
            match = re.search(r'(figure|fig\.?|table)\s+(\d+)', caption["text"], re.IGNORECASE)
            if not match:
                continue
                
            fig_type = match.group(1).lower()
            fig_num = match.group(2)
            
            # Estimate figure region - usually above the caption
            caption_bbox = caption["bbox"]  # [x0, y0, x1, y1]
            
            # For figures, look above the caption
            if fig_type in ["figure", "fig"]:
                # Start with a reasonable region above the caption
                fig_bbox = [
                    caption_bbox[0] - 20,  # x0: slightly wider than caption
                    caption_bbox[1] - 300,  # y0: look up to 300 points above
                    caption_bbox[2] + 20,  # x1: slightly wider than caption
                    caption_bbox[1] - 10   # y1: stop just above caption
                ]
                
                # Make sure we don't go beyond page boundaries
                fig_bbox[0] = max(0, fig_bbox[0])
                fig_bbox[1] = max(0, fig_bbox[1])
                fig_bbox[2] = min(page.rect.width, fig_bbox[2])
                fig_bbox[3] = max(fig_bbox[1] + 100, min(caption_bbox[1], fig_bbox[3]))  # Ensure minimum height
            
            # For tables, look below the caption
            else:
                # Start with a reasonable region below the caption
                fig_bbox = [
                    caption_bbox[0] - 20,  # x0: slightly wider than caption
                    caption_bbox[3] + 10,  # y0: start just below caption
                    caption_bbox[2] + 20,  # x1: slightly wider than caption
                    caption_bbox[3] + 300  # y1: look up to 300 points below
                ]
                
                # Make sure we don't go beyond page boundaries
                fig_bbox[0] = max(0, fig_bbox[0])
                fig_bbox[1] = min(page.rect.height, fig_bbox[1])
                fig_bbox[2] = min(page.rect.width, fig_bbox[2])
                fig_bbox[3] = min(page.rect.height, fig_bbox[3])
            
            # Extract the figure region
            clip_rect = fitz.Rect(fig_bbox)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip_rect)
            img_bytes = pixmap.tobytes("png")
            
            # Generate unique ID
            fig_id = f"{pdf_name}_p{page_num}_{fig_type}{fig_num}"
            
            # Save image if requested
            img_path = None
            if output_dir:
                img_path = os.path.join(output_dir, f"{fig_id}.png")
                with open(img_path, "wb") as f:
                    f.write(img_bytes)
            
            # Add to our list
            figure_data = {
                "id": fig_id,
                "document_name": pdf_name,
                "page_number": page_num + 1,
                "figure_type": fig_type,
                "figure_number": fig_num,
                "caption": caption["text"],
                "bbox": fig_bbox,
                "format": "png",
                "path": img_path,
                "image_bytes": img_bytes,
                "content_type": "image",
                "is_figure": True
            }
            
            figure_list.append(figure_data)
    
    except Exception as e:
        logger.error(f"Error detecting figures on page {page_num}: {e}")
    
    return figure_list

def main():
    """Main function to run the script"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Extract text and images from PDFs and save embeddings to Pinecone")
    parser.add_argument("--pdf_dir", type=str, default="clinical_files",
                        help="Directory containing PDF files")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Batch size for uploading vectors to Pinecone")
    parser.add_argument("--test", action="store_true",
                        help="Run in test mode (only extract text, don't upload to Pinecone)")
    parser.add_argument("--save_markdown", action="store_true",
                        help="Save the markdown generated by docling to files")
    parser.add_argument("--save_paragraphs", type=str, choices=["md", "json", "txt"],
                        help="Save paragraphs in the specified format (md, json, txt)")
    parser.add_argument("--context_window_size", type=int, default=2,
                        help="Number of paragraphs before and after to include as context (default: 2)")
    parser.add_argument("--overlap", type=int, default=1, 
                        help="Number of paragraphs to overlap between chunks (default: 1)")
    parser.add_argument("--extract_images", action="store_true",
                        help="Extract and process images from PDFs")
    parser.add_argument("--save_images_dir", type=str, default="extracted_images",
                        help="Directory to save extracted images")
    parser.add_argument("--pdf_limit", type=int, default=None,
                        help="Maximum number of PDFs to process (None for all)")
    parser.add_argument("--specific_pdf", type=str, default=None,
                        help="Specific PDF filename to process (None for all)")
    args = parser.parse_args()
    
    if args.test:
        # Test mode - just extract text from PDFs
        pdf_dir_path = Path(args.pdf_dir)
        
        pdf_files = list(pdf_dir_path.glob("**/*.pdf"))
        
        if not pdf_files:
            logger.info(f"No PDF files found in {args.pdf_dir}")
            return
        
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        # Process first PDF file as a test
        if pdf_files:
            test_file = pdf_files[0]
            logger.info(f"Testing extraction on: {test_file}")
            try:
                text = extract_text_from_pdf(str(test_file), save_markdown=args.save_markdown)
                paragraphs = chunk_text_by_paragraphs(text, overlap=args.overlap)
                logger.info(f"Successfully extracted {len(paragraphs)} paragraphs with {args.overlap} paragraph overlap")
                
                # Save paragraphs if requested
                if args.save_paragraphs:
                    output_path = test_file.with_name(f"{test_file.stem}_paragraphs")
                    save_paragraphs(paragraphs, output_path, format=args.save_paragraphs)
                
                logger.info("\nSample paragraphs:")
                for i, p in enumerate(paragraphs[:3]):  # Show first 3 paragraphs
                    logger.info(f"\nParagraph {i+1}:\n{p[:200]}...")
                
                # Test image extraction if requested
                if args.extract_images:
                    logger.info(f"Testing image extraction on: {test_file}")
                    try:
                        img_dir = os.path.join(args.save_images_dir, test_file.stem) if args.save_images_dir else None
                        images = extract_images_from_pdf(str(test_file), output_dir=img_dir)
                        logger.info(f"Successfully extracted {len(images)} images")
                        
                        # Test image classification and description for the first image
                        if images:
                            first_img = images[0]
                            image_type = classify_research_image(first_img["image_bytes"], first_img.get("caption"))
                            description = get_image_description(first_img["image_bytes"], first_img.get("caption"))
                            logger.info(f"\nImage type: {image_type}")
                            logger.info(f"\nImage description sample:\n{description[:300]}...")
                    except Exception as e:
                        logger.error(f"Error during test image extraction: {e}")
            except Exception as e:
                logger.error(f"Error during test extraction: {e}")
    else:
        # Normal mode - process PDFs and upload to Pinecone
        process_pdf_directory(args.pdf_dir, args.batch_size, 
                             save_markdown=args.save_markdown,
                             save_paragraphs_format=args.save_paragraphs,
                             context_window_size=args.context_window_size,
                             overlap=args.overlap,
                             extract_images=args.extract_images,
                             save_images_dir=args.save_images_dir,
                             pdf_limit=args.pdf_limit,
                             specific_pdf=args.specific_pdf)
    
        logger.info("Processing complete!")

if __name__ == "__main__":
    main()
