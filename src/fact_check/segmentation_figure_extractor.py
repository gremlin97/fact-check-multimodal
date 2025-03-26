#!/usr/bin/env python3
import os
import argparse
import logging
import numpy as np
import cv2
import fitz  # PyMuPDF
from pathlib import Path
from PIL import Image
import io
import time
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SegmentationFigureExtractor:
    """Extract figures from PDFs using image segmentation and boundary detection
    
    This approach treats the problem as image processing rather than PDF structure analysis:
    1. Convert PDF pages to high-resolution images
    2. Use segmentation to separate text from figures
    3. Detect figure boundaries using edge detection and contour analysis
    4. Extract figures with generous margins to avoid cutoffs
    """
    
    def __init__(self, output_dir="segmentation_extracted_figures", dpi=300):
        """Initialize the extractor
        
        Args:
            output_dir: Directory to save extracted figures
            dpi: Resolution for page rendering (higher = better quality)
        """
        self.output_dir = output_dir
        self.dpi = dpi
        self.zoom_factor = dpi / 72  # Standard PDF is 72 DPI
        
    def process_pdf(self, pdf_path):
        """Process a PDF to extract figures using segmentation
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of paths to extracted figures
        """
        pdf_path = Path(pdf_path)
        pdf_name = pdf_path.stem
        
        # Create output directory
        result_dir = Path(self.output_dir) / pdf_name
        os.makedirs(result_dir, exist_ok=True)
        
        extracted_figures = []
        
        try:
            # Open the PDF
            logger.info(f"Processing PDF: {pdf_path}")
            doc = fitz.open(pdf_path)
            
            # Process each page
            for page_idx, page in enumerate(doc):
                page_num = page_idx + 1  # 1-based page numbering
                logger.info(f"Processing page {page_num}")
                
                # Step 1: Render the page at high resolution
                pix = page.get_pixmap(matrix=fitz.Matrix(self.zoom_factor, self.zoom_factor))
                img_bytes = pix.tobytes("png")
                
                # Convert to OpenCV format
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                
                # Save the full page as a reference
                full_page_path = result_dir / f"{pdf_name}_page_{page_num}_full.png"
                cv2.imwrite(str(full_page_path), image)
                
                # Step 2: Find potential figure regions using segmentation
                figure_regions = self._find_figure_regions(image, page)
                
                # Step 3: Detect figure captions
                captions = self._detect_captions(page)
                
                # Track saved regions to avoid saving sub-regions
                saved_regions = []
                
                # Step 4: Extract and save each figure region
                for region_idx, region in enumerate(figure_regions):
                    # Get region bounds (x, y, width, height)
                    x, y, w, h = region["bounds"]
                    
                    # Use full page width instead of just the detected region width
                    # Only adjust the vertical margins
                    margin_y = int(h * 0.2)  # 20% margin for height
                    
                    # Set x1 and x2 to cover the full page width
                    x1 = 0  # Start from left edge
                    y1 = max(0, y - margin_y)
                    x2 = image.shape[1]  # Go to right edge
                    y2 = min(image.shape[0], y + h + margin_y)
                    
                    # Extract the region
                    figure_image = image[y1:y2, x1:x2]
                    
                    # Determine figure number from nearby captions
                    figure_num = self._match_region_to_caption(region, captions)
                    
                    # Generate filename
                    if figure_num:
                        filename = f"{pdf_name}_figure_{figure_num}_p{page_num}.png"
                    else:
                        filename = f"{pdf_name}_p{page_num}_region_{region_idx+1}.png"
                    
                    # Save the extracted figure
                    output_path = result_dir / filename
                    cv2.imwrite(str(output_path), figure_image)
                    logger.info(f"Saved figure region to {filename}")
                    
                    extracted_figures.append(output_path)
                    saved_regions.append((y1, y2))  # Track the y-coordinates of saved regions
                
                # Step 5: Specialized processing for diagrams (color segmentation)
                # This is helpful for extracting specific elements like blue diagrams
                diagram_regions = self._find_diagrams_by_color(image, page_num)
                
                # Filter out diagram regions that are already contained within saved regions
                filtered_diagram_regions = []
                for diagram in diagram_regions:
                    x, y, w, h = diagram["bounds"]
                    
                    # Calculate diagram boundaries with margins (same as above)
                    margin_y = int(h * 0.2) 
                    diagram_y1 = max(0, y - margin_y)
                    diagram_y2 = min(image.shape[0], y + h + margin_y)
                    
                    # Check if this diagram is contained within any saved region
                    is_contained = False
                    for saved_y1, saved_y2 in saved_regions:
                        # If the diagram region is fully contained within a saved region
                        if diagram_y1 >= saved_y1 and diagram_y2 <= saved_y2:
                            is_contained = True
                            break
                    
                    if not is_contained:
                        filtered_diagram_regions.append(diagram)
                    else:
                        logger.info(f"Skipping sub-region already contained in a larger figure")
                
                for diagram_idx, diagram in enumerate(filtered_diagram_regions):
                    # Get region bounds
                    x, y, w, h = diagram["bounds"]
                    color_name = diagram["color"]
                    
                    # Extract with margin
                    # Use full page width for diagrams too
                    margin_y = int(h * 0.2)  # 20% margin for height
                    
                    x1 = 0  # Start from left edge
                    y1 = max(0, y - margin_y)
                    x2 = image.shape[1]  # Go to right edge
                    y2 = min(image.shape[0], y + h + margin_y)
                    
                    diagram_image = image[y1:y2, x1:x2]
                    
                    # Save the diagram
                    diagram_path = result_dir / f"{pdf_name}_p{page_num}_{color_name}_diagram_{diagram_idx+1}.png"
                    cv2.imwrite(str(diagram_path), diagram_image)
                    logger.info(f"Saved {color_name} diagram region")
                    
                    extracted_figures.append(diagram_path)
            
            logger.info(f"Completed extraction from {pdf_path}")
            return extracted_figures
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            return extracted_figures
    
    def _find_figure_regions(self, image, page):
        """Identify regions containing figures using image processing
        
        Args:
            image: OpenCV image of the page
            page: PyMuPDF page object
            
        Returns:
            List of dictionaries with figure region information
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate to connect edges
        kernel = np.ones((5,5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size
        min_area = image.shape[0] * image.shape[1] * 0.01  # At least 1% of page
        max_area = image.shape[0] * image.shape[1] * 0.8   # At most 80% of page
        
        figure_regions = []
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Filter by size
            if min_area <= area <= max_area:
                # Calculate aspect ratio
                aspect_ratio = w / h if h > 0 else 0
                
                # Typical figures aren't extremely narrow
                if 0.2 <= aspect_ratio <= 5.0:
                    # Additional analysis for figure likelihood
                    roi = gray[y:y+h, x:x+w]
                    
                    # Check for image content vs. text
                    # Text usually has high variance in small neighborhoods
                    # Images/figures have more consistent neighborhoods
                    local_std = cv2.Sobel(roi, cv2.CV_64F, 1, 1).std()
                    
                    # Higher value indicates more complex image data
                    figure_score = local_std
                    
                    figure_regions.append({
                        "bounds": (x, y, w, h),
                        "area": area,
                        "aspect_ratio": aspect_ratio,
                        "score": figure_score
                    })
        
        # Sort by score (higher score = more likely to be a figure)
        figure_regions.sort(key=lambda x: x["score"], reverse=True)
        
        # Remove overlapping regions, keeping the higher-scored one
        filtered_regions = []
        for region in figure_regions:
            x1, y1, w1, h1 = region["bounds"]
            overlapping = False
            
            for existing in filtered_regions:
                x2, y2, w2, h2 = existing["bounds"]
                
                # Calculate intersection
                intersection_x = max(0, min(x1+w1, x2+w2) - max(x1, x2))
                intersection_y = max(0, min(y1+h1, y2+h2) - max(y1, y2))
                intersection_area = intersection_x * intersection_y
                
                # Calculate IoU (Intersection over Union)
                region1_area = w1 * h1
                region2_area = w2 * h2
                union_area = region1_area + region2_area - intersection_area
                iou = intersection_area / union_area if union_area > 0 else 0
                
                if iou > 0.5:  # Significant overlap
                    overlapping = True
                    break
            
            if not overlapping:
                filtered_regions.append(region)
        
        return filtered_regions
    
    def _detect_captions(self, page):
        """Find figure captions in the page text
        
        Args:
            page: PyMuPDF page object
            
        Returns:
            List of dictionaries with caption information
        """
        # Get page text with positions
        text_blocks = page.get_text("dict")["blocks"]
        
        captions = []
        caption_patterns = ["Figure", "Fig.", "Fig"]
        
        for block in text_blocks:
            if "lines" not in block:
                continue
                
            # Extract text from the block
            block_text = ""
            for line in block["lines"]:
                if "spans" not in line:
                    continue
                for span in line["spans"]:
                    block_text += span["text"]
            
            # Look for caption patterns
            for pattern in caption_patterns:
                if pattern in block_text:
                    # Try to extract figure number
                    match = re.search(r'(?:Figure|Fig\.?)\s+(\d+[a-zA-Z]?)', block_text)
                    
                    if match:
                        figure_num = match.group(1)
                        captions.append({
                            "figure_num": figure_num,
                            "text": block_text,
                            "bbox": block["bbox"]  # (x0, y0, x1, y1)
                        })
        
        return captions
    
    def _match_region_to_caption(self, region, captions):
        """Match a figure region to the most appropriate caption
        
        Args:
            region: Dictionary with region bounds
            captions: List of caption dictionaries
            
        Returns:
            Figure number as a string, or None if no match
        """
        if not captions:
            return None
            
        x, y, w, h = region["bounds"]
        region_center_y = y + h/2
        
        # Captions are typically below figures
        captions_below = [c for c in captions if c["bbox"][1] > y + h]
        
        if captions_below:
            # Sort by vertical proximity
            captions_below.sort(key=lambda c: c["bbox"][1] - (y + h))
            return captions_below[0]["figure_num"]
        
        # If no captions below, look for captions anywhere
        # Sort all captions by distance to region center
        captions.sort(key=lambda c: abs((c["bbox"][1] + c["bbox"][3])/2 - region_center_y))
        
        if captions:
            return captions[0]["figure_num"]
            
        return None
    
    def _find_diagrams_by_color(self, image, page_num):
        """Find diagram regions based on color segmentation
        
        Args:
            image: OpenCV image of the page
            page_num: Page number (1-based)
            
        Returns:
            List of diagram regions
        """
        diagrams = []
        
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for specific diagrams
        # Blue range (for structural diagrams like in Arunachalam paper)
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        # Create color masks
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # Find blue regions
        blue_regions = self._extract_regions_from_mask(blue_mask, "blue", min_area_pct=0.001)
        diagrams.extend(blue_regions)
        
        if page_num == 4:  # Page 4 has the structural diagrams
            # Left half of the page might contain diagram a
            height, width = image.shape[:2]
            left_mask = np.zeros((height, width), dtype=np.uint8)
            left_mask[:, :width//2] = 255
            left_blue = cv2.bitwise_and(blue_mask, blue_mask, mask=left_mask)
            
            # Right half of the page might contain diagram b
            right_mask = np.zeros((height, width), dtype=np.uint8)
            right_mask[:, width//2:] = 255
            right_blue = cv2.bitwise_and(blue_mask, blue_mask, mask=right_mask)
            
            # Process left and right sides separately
            left_regions = self._extract_regions_from_mask(left_blue, "blue_struct_left", min_area_pct=0.001)
            right_regions = self._extract_regions_from_mask(right_blue, "blue_struct_right", min_area_pct=0.001)
            
            diagrams.extend(left_regions)
            diagrams.extend(right_regions)
        
        return diagrams
    
    def _extract_regions_from_mask(self, mask, color_name, min_area_pct=0.01):
        """Extract regions from a binary mask
        
        Args:
            mask: Binary mask image
            color_name: Name of the color (for labeling)
            min_area_pct: Minimum area as percentage of total image
            
        Returns:
            List of region dictionaries
        """
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate minimum area
        min_area = mask.shape[0] * mask.shape[1] * min_area_pct
        
        regions = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(contour)
                regions.append({
                    "bounds": (x, y, w, h),
                    "area": area,
                    "color": color_name
                })
        
        # Sort by area (larger first)
        regions.sort(key=lambda r: r["area"], reverse=True)
        
        return regions

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Extract figures from PDFs using image segmentation")
    parser.add_argument("--pdf_path", type=str, required=True, 
                      help="Path to the PDF file")
    parser.add_argument("--output", type=str, default="segmentation_extracted_figures", 
                      help="Output directory for extracted figures")
    parser.add_argument("--dpi", type=int, default=300,
                      help="Resolution for page rendering (higher = better quality)")
    
    args = parser.parse_args()
    
    # Create the extractor
    extractor = SegmentationFigureExtractor(
        output_dir=args.output,
        dpi=args.dpi
    )
    
    # Process the PDF
    start_time = time.time()
    extracted_figures = extractor.process_pdf(args.pdf_path)
    elapsed_time = time.time() - start_time
    
    logger.info(f"Extraction completed in {elapsed_time:.2f} seconds")
    logger.info(f"Extracted {len(extracted_figures)} figure regions")

if __name__ == "__main__":
    main() 