import os
import subprocess
import time
import base64
import logging
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec
from PIL import Image
import io

# Set up argument parser
parser = argparse.ArgumentParser(description="Process PDFs for text and image extraction and indexing")
parser.add_argument("--start_index", type=int, default=0, help="PDF index to start from (0-based)")
parser.add_argument("--processed_file", type=str, default="processed_pdfs.txt", 
                   help="File to track processed PDFs for resuming later")
parser.add_argument("--skip_processed", action="store_true", 
                   help="Skip PDFs that have already been processed according to the processed file")
parser.add_argument("--verbose", action="store_true", help="Enable more detailed logging")
args = parser.parse_args()

# Set up logging
log_level = logging.DEBUG if args.verbose else logging.INFO
logging.basicConfig(
    level=log_level, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Log to console
        logging.FileHandler("pdf_processing.log")  # Log to file
    ]
)
logger = logging.getLogger(__name__)

# Create a console handler for prominent status messages
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console_formatter = logging.Formatter('>>> %(message)s')
console.setFormatter(console_formatter)

status_logger = logging.getLogger("status")
status_logger.setLevel(logging.INFO)
status_logger.addHandler(console)
status_logger.propagate = False

def log_status(message):
    """Log a prominent status message"""
    status_logger.info(message)
    # Also log to the regular logger
    logger.info(message)

log_status("Starting PDF processing pipeline")

# Load environment variables
load_dotenv()

# Configure API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GOOGLE_API_KEY or not PINECONE_API_KEY:
    raise ValueError("Missing required API keys. Please set GOOGLE_API_KEY and PINECONE_API_KEY in .env file")

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

# Configure Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "med-cite-index"

# Function to load already processed PDFs
def load_processed_pdfs(filename):
    processed = set()
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            for line in f:
                processed.add(line.strip())
    return processed

# Function to save a PDF as processed
def mark_as_processed(filename, pdf_path):
    with open(filename, 'a') as f:
        f.write(f"{pdf_path}\n")

# Load processed PDFs if requested
processed_pdfs = set()
if args.skip_processed:
    processed_pdfs = load_processed_pdfs(args.processed_file)
    log_status(f"Found {len(processed_pdfs)} already processed PDFs that will be skipped")

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
            
        return pc.Index(INDEX_NAME)
    except Exception as e:
        logger.error(f"Error with Pinecone: {e}")
        raise

def resize_image(image_path, max_size=(800, 800), quality=85):
    """Resize image to reduce file size
    
    Args:
        image_path: Path to the image file
        max_size: Maximum dimensions (width, height)
        quality: JPEG quality (0-100)
        
    Returns:
        Image bytes
    """
    try:
        # Open image
        img = Image.open(image_path)
        
        # Resize if needed
        if img.width > max_size[0] or img.height > max_size[1]:
            img.thumbnail(max_size)
        
        # Convert to JPEG
        output = io.BytesIO()
        if img.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        img.save(output, format='JPEG', quality=quality)
        
        return output.getvalue()
    except Exception as e:
        logger.error(f"Error resizing image: {e}")
        return None

def get_image_description(image_path):
    """Generate a detailed description for an image using Gemini Vision
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Detailed description text
    """
    try:
        # Resize image
        image_data = resize_image(image_path)
        if not image_data:
            return "Error: Could not resize image"
        
        # Encode image as base64
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Set up Gemini 1.5 Flash model with correct model name
        model = genai.GenerativeModel('gemini-1.5-flash-002')  # Updated to the latest version
        
        # Craft more detailed prompt for research figure analysis
        prompt = """
        Provide an extremely detailed and comprehensive analysis of this research paper figure or diagram.
        
        Describe:
        1. The figure type (graph, diagram, chart, flowchart, molecular structure, etc.)
        2. All visual elements present (axes, labels, legends, arrows, symbols, structures)
        3. Colors, patterns, and visual organization
        4. Data representation (e.g., bars, lines, points, error bars)
        5. All text content visible in the image, including axis labels, titles, annotations
        6. All numerical values that can be read from the figure
        7. Key relationships, patterns, or trends shown
        8. Statistical significance indicators if present
        9. Any conclusions or findings that can be drawn
        10. How this figure might relate to research in this field
        
        Be extremely detailed and specific. Don't miss any visual information, text, numbers or patterns.
        Avoid vague or general statements. Include all facts and data points visible in the image.
        """
        
        # Configure generation parameters for more detailed output
        generation_config = {
            "temperature": 0.2,  # Lower temperature for more factual output
            "max_output_tokens": 2048,  # Allow longer output for detailed descriptions
            "top_p": 0.95,
            "top_k": 40
        }
        
        # Generate response with updated format and config
        response = model.generate_content(
            contents=[
                {
                    "parts": [
                        {"text": prompt},
                        {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}}
                    ]
                }
            ],
            generation_config=generation_config
        )
        
        return response.text
    except Exception as e:
        logger.error(f"Error generating image description: {e}")
        return "Error: Could not generate image description"

def upload_images_to_pinecone(image_dir, pdf_name, batch_size=10):
    """Process extracted images and upload to Pinecone
    
    Args:
        image_dir: Directory containing extracted images
        pdf_name: Name of the PDF file
        batch_size: Batch size for Pinecone uploads
    
    Returns:
        Number of images processed
    """
    try:
        # Set up Pinecone index
        log_status(f"Setting up Pinecone index for image vectors from {pdf_name}")
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
        
        # Process images
        batch = []
        image_count = 0
        
        # Find all PNG images in the directory
        image_files = list(Path(image_dir).glob("*.png"))
        log_status(f"Found {len(image_files)} images in {image_dir}")
        
        # Filter out full page images
        filtered_image_files = [img for img in image_files if "full" not in img.name]
        log_status(f"Processing {len(filtered_image_files)} figures after filtering out full page images")
        
        # Sort files to process them in a consistent order
        filtered_image_files.sort()
        
        for image_file in filtered_image_files:
            try:
                image_path = str(image_file)
                image_name = image_file.name
                
                # Skip any other non-figure images
                if "page" in image_name.lower() and "full" in image_name.lower():
                    continue
                
                # Generate detailed description
                logger.info(f"Generating detailed description for {image_name}")
                description = get_image_description(image_path)
                logger.info(f"Description length: {len(description)} characters")
                
                # Get embeddings for description using the same function as text content
                logger.info(f"Generating embeddings for description")
                embeddings = get_embeddings(description)  # Using imported get_embeddings from extract.py
                
                if not embeddings:
                    logger.warning(f"Failed to generate embeddings for {image_name}")
                    continue
                
                # Create metadata
                metadata = {
                    "document_name": pdf_name,
                    "image_name": image_name,
                    "image_path": image_path,
                    "description": description,
                    "embedding_type": "text",
                    "content_type": "image_description"
                }
                
                # Create vector ID
                vector_id = f"{pdf_name}_{image_file.stem}"
                
                # Create vector
                vector = {
                    "id": vector_id,
                    "values": embeddings["dense"],
                    "metadata": metadata
                }
                
                # Add sparse values if available and supported
                if supports_sparse and "sparse" in embeddings:
                    vector["sparse_values"] = embeddings["sparse"]
                
                # Add to batch
                batch.append(vector)
                image_count += 1
                logger.info(f"Added vector for {image_name} to batch (now {len(batch)} items)")
                
                # Upload in batches
                if len(batch) >= batch_size:
                    log_status(f"Uploading batch of {len(batch)} image vectors to Pinecone")
                    start_time = time.time()
                    index.upsert(vectors=batch)
                    logger.info(f"Batch upload complete in {time.time() - start_time:.2f}s")
                    batch = []
            
            except Exception as e:
                logger.error(f"Error processing image {image_file}: {e}")
        
        # Upload any remaining vectors
        if batch:
            log_status(f"Uploading final batch of {len(batch)} image vectors to Pinecone")
            start_time = time.time()
            index.upsert(vectors=batch)
            logger.info(f"Final batch upload complete in {time.time() - start_time:.2f}s")
        
        return image_count
    
    except Exception as e:
        logger.error(f"Error uploading images to Pinecone: {e}")
        return 0

# Directory containing PDFs
pdf_dir = "src/fact_check/clinical_files"

# Output directory for images
output_base_dir = "extracted_figures_segmentation"

# Get list of all PDFs
pdf_files = list(Path(pdf_dir).glob("**/*.pdf"))
log_status(f"Found {len(pdf_files)} PDF files in {pdf_dir}")

# Validate start index
if args.start_index < 0 or args.start_index >= len(pdf_files):
    log_status(f"Warning: Start index {args.start_index} is out of range, adjusting to 0")
    start_index = 0
else:
    start_index = args.start_index
    log_status(f"Starting from PDF index {start_index} ({pdf_files[start_index].name})")

# Process each PDF individually, starting from the specified index
for i, pdf_file in enumerate(pdf_files[start_index:], start=start_index):
    pdf_name = pdf_file.name
    pdf_stem = pdf_file.stem
    output_dir = f"{output_base_dir}/{pdf_stem}"
    
    # Check if already processed
    if args.skip_processed and str(pdf_file) in processed_pdfs:
        log_status(f"\nSkipping already processed PDF {i+1}/{len(pdf_files)}: {pdf_name}")
        continue
    
    log_status(f"\n\n{'='*80}")
    log_status(f"Processing PDF {i+1}/{len(pdf_files)}: {pdf_name}")
    log_status(f"{'='*80}\n")
    
    try:
        # Step 1: Process text content following extract.py pattern
        log_status(f"STEP 1: Processing text content from {pdf_name}...")
        # Import needed functions from extract.py
        from src.fact_check.extract import extract_text_from_pdf, chunk_text_by_paragraphs, get_embeddings, detect_and_extract_figures
        
        # Extract text from PDF using docling (will fallback to PyPDF if needed)
        log_status(f"Extracting text from {pdf_name}...")
        start_time = time.time()
        text = extract_text_from_pdf(str(pdf_file), save_markdown=False)
        log_status(f"Text extraction completed in {time.time() - start_time:.2f}s")
        
        if not text or len(text.strip()) < 100:
            logger.warning(f"Warning: Extracted text is very short or empty for {pdf_name}. This may indicate an extraction issue.")
        
        # Chunk text into paragraphs with overlap
        overlap = 1  # Number of paragraphs to overlap
        max_length = 2500  # Maximum length of paragraphs
        min_length = 1000  # Minimum preferred length
        
        log_status(f"Chunking text into paragraphs...")
        start_time = time.time()
        paragraphs = chunk_text_by_paragraphs(text, max_length=max_length, min_length=min_length, overlap=overlap)
        log_status(f"Created {len(paragraphs)} paragraphs with {overlap} overlap in {time.time() - start_time:.2f}s")
        
        # Set up Pinecone index for text content
        index = create_pinecone_index_if_not_exists()
        
        # Process each paragraph
        text_batch = []
        text_count = 0
        batch_size = 10
        context_window_size = 2  # Number of paragraphs before and after to include as context
        
        # Check if the existing index supports sparse vectors
        try:
            index_info = pc.describe_index(INDEX_NAME)
            supports_sparse = hasattr(index_info, 'metric') and index_info.metric == 'dotproduct'
            if not supports_sparse:
                logger.warning("Warning: Index does not support sparse vectors. Only using dense vectors.")
        except Exception:
            supports_sparse = False
            logger.warning("Warning: Could not determine if index supports sparse vectors. Only using dense vectors.")
        
        log_status(f"Processing {len(paragraphs)} paragraphs...")
        for i, paragraph in enumerate(paragraphs):
            if i % 10 == 0:  # Log progress every 10 paragraphs
                log_status(f"  Processing paragraph {i+1}/{len(paragraphs)}")
            
            if not paragraph or len(paragraph.strip()) < 50:  # Skip empty or very short paragraphs
                logger.debug(f"Skipping empty or very short paragraph {i}")
                continue
            
            # Determine if paragraph is a header
            is_header = paragraph.startswith('#')
            header_level = 0
            if is_header:
                import re
                header_level = len(re.match(r'^#+', paragraph).group())
            
            # Create metadata with overlap information
            metadata = {
                "document_name": pdf_name,
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
            try:
                logger.debug(f"Getting embeddings for paragraph {i} with {len(context_paragraphs)} context paragraphs")
                embeddings = get_embeddings(paragraph, context_paragraphs)
                
                if not embeddings:
                    logger.warning(f"Failed to get embeddings for paragraph {i} in {pdf_name}")
                    continue
                    
                # Create vector ID
                vector_id = f"{pdf_stem}_p{i}"
                
                # Ensure no null values in metadata (Pinecone requirement)
                for key, value in list(metadata.items()):
                    if value is None:
                        metadata[key] = ""  # Replace None with empty string
                
                # Create vector
                vector = {
                    "id": vector_id,
                    "values": embeddings["dense"],
                    "metadata": metadata
                }
                
                # Add sparse values if available and supported
                if supports_sparse and "sparse" in embeddings:
                    vector["sparse_values"] = embeddings["sparse"]
                
                # Add to batch
                text_batch.append(vector)
                text_count += 1
                logger.debug(f"Added vector for paragraph {i} to batch (now {len(text_batch)} items)")
                
            except Exception as e:
                logger.error(f"Error processing paragraph {i}: {e}")
                continue
            
            # Upload in batches
            if len(text_batch) >= batch_size:
                log_status(f"Uploading batch of {len(text_batch)} text vectors to Pinecone")
                start_time = time.time()
                index.upsert(vectors=text_batch)
                logger.info(f"Batch upload completed in {time.time() - start_time:.2f}s")
                text_batch = []
        
        # Upload any remaining text vectors
        if text_batch:
            log_status(f"Uploading final batch of {len(text_batch)} text vectors to Pinecone")
            start_time = time.time()
            index.upsert(vectors=text_batch)
            logger.info(f"Final text batch upload completed in {time.time() - start_time:.2f}s")
        
        log_status(f"Processed {text_count} text chunks from {pdf_name}")
        
        # Step 2: Extract figures using segmentation-based extractor
        log_status(f"\nSTEP 2: Extracting figures from {pdf_name}...")
        cmd = f"python segmentation_figure_extractor.py --pdf_path \"{pdf_file}\" --output \"{output_base_dir}\" --dpi 400"
        
        # Run command with timeout to prevent hanging on problematic PDFs
        try:
            start_time = time.time()
            # Log command for debugging
            logger.debug(f"Running command: {cmd}")
            
            # Use a timeout of 120 seconds to prevent hanging
            result = subprocess.run(cmd, shell=True, check=False, timeout=120)
            elapsed = time.time() - start_time
            
            if result.returncode != 0:
                log_status(f"\nWarning: Extraction command exited with non-zero code: {result.returncode}")
                log_status(f"Will still attempt to process any extracted figures...")
            
            log_status(f"\nExtraction completed in {elapsed:.2f} seconds")
            
            # Step 3: Upload images to Pinecone
            log_status(f"STEP 3: Uploading figures to Pinecone...")
            image_dir = f"{output_base_dir}/{pdf_stem}"
            
            # Check if directory exists and contains files before proceeding
            if not os.path.exists(image_dir):
                logger.warning(f"Warning: Output directory {image_dir} does not exist. Skipping upload.")
                continue
                
            # Check if the directory contains any PNG files
            png_files = list(Path(image_dir).glob("*.png"))
            if not png_files:
                logger.warning(f"Warning: No PNG files found in {image_dir}. Skipping upload.")
                continue
                
            # Log all found PNG files in verbose mode
            if args.verbose:
                logger.debug("Found PNG files:")
                for png_file in png_files:
                    logger.debug(f"  - {png_file}")
                
            # If we have files, proceed with upload regardless of extractor exit code
            log_status(f"Found {len(png_files)} PNG files to process")
            image_count = upload_images_to_pinecone(image_dir, pdf_name)
            log_status(f"Successfully uploaded {image_count} images to Pinecone")
            
        except subprocess.TimeoutExpired:
            logger.error(f"\nTimeout error processing {pdf_name}: Command took too long to complete")
        except subprocess.CalledProcessError as e:
            logger.error(f"\nError processing {pdf_name}: {e}")
        except Exception as e:
            logger.error(f"\nUnexpected error processing {pdf_name}: {e}")
        
        # Mark as processed after successful processing
        mark_as_processed(args.processed_file, str(pdf_file))
        log_status(f"Marked {pdf_name} as processed in {args.processed_file}")
        
        # Sleep for a moment between PDFs
        log_status(f"Waiting 5 seconds before processing next PDF...")
        time.sleep(5)

    except Exception as e:
        logger.error(f"\nUnexpected error processing {pdf_name}: {e}")
        # Don't mark as processed if there was an error

log_status("\nAll PDFs have been processed!") 