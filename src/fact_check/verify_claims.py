import os
import json
import logging
import argparse
import time
from typing import List, Dict, Any
import google.generativeai as genai
from pinecone import Pinecone
from dotenv import load_dotenv
from tqdm import tqdm
from pinecone_text.sparse import BM25Encoder
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure API keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Configure Google Gemini
genai.configure(api_key=GOOGLE_API_KEY)

# Configure Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "med-cite-index"

# Rate limiting for Gemini Pro model
LAST_API_CALL_TIME = 0
MIN_TIME_BETWEEN_CALLS = 60  # seconds

def rate_limited_api_call(func):
    """Decorator to rate limit API calls"""
    def wrapper(*args, **kwargs):
        global LAST_API_CALL_TIME
        
        # Check if we need to wait
        current_time = time.time()
        time_since_last_call = current_time - LAST_API_CALL_TIME
        
        if time_since_last_call < MIN_TIME_BETWEEN_CALLS:
            wait_time = MIN_TIME_BETWEEN_CALLS - time_since_last_call
            logger.info(f"Rate limiting: Waiting {wait_time:.1f} seconds before next API call to respect quota limits...")
            time.sleep(wait_time)
        
        # Make the API call
        result = func(*args, **kwargs)
        
        # Update the last call time
        LAST_API_CALL_TIME = time.time()
        
        return result
    
    return wrapper

def load_claims(claims_file: str) -> List[Dict[str, str]]:
    """Load claims from a JSON file"""
    with open(claims_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("claims", [])

def preprocess_claim(claim: str) -> str:
    """Preprocess the claim by expanding or rephrasing it for better retrieval"""
    # Create a more neutral prompt without full marketing context
    return f"In medical literature, is it true that {claim}?"

def get_embedding(text: str) -> Dict[str, Any]:
    """Get dense and sparse embeddings for a text using Gemini and BM25"""
    try:
        # Get dense embedding
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_query"
        )
        dense_embedding = result["embedding"]
        
        # Generate sparse embedding
        try:
            # Initialize BM25 encoder with default parameters
            bm25 = BM25Encoder.default()
            
            # Encode the text as a sparse vector for queries
            sparse_vector = bm25.encode_queries(text)
            
            # Return both dense and sparse embeddings
            return {
                "dense": dense_embedding,
                "sparse": sparse_vector
            }
        except ImportError:
            logger.info("Warning: pinecone_text not installed. Falling back to dense-only search.")
    
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")

def search_evidence(claim: str, top_k: int = 5, max_duplicates_per_source: int = 2) -> List[Dict[str, Any]]:
    """Search for evidence supporting a claim in the Pinecone index using hybrid search"""
    try:
        # Get embeddings for the claim
        embeddings = get_embedding(claim)
        
        # Prepare query parameters
        query_params = {
            "vector": embeddings["dense"],
            "top_k": top_k * 3,  # Retrieve more initially to ensure diversity
            "include_metadata": True
        }
        
        # Add sparse vector if available for hybrid search
        if embeddings["sparse"] is not None:
            query_params["sparse_vector"] = embeddings["sparse"]
        
        # Query Pinecone index
        index = pc.Index(INDEX_NAME)
        results = index.query(**query_params)
        
        # Add detailed logging of all returned matches
        logger.info(f"Total matches found: {len(results.matches)}")
        doc_counts_total = {}
        for i, match in enumerate(results.matches):
            doc_name = match.metadata.get("document_name", "Unknown")
            doc_counts_total[doc_name] = doc_counts_total.get(doc_name, 0) + 1
        logger.info(f"Sources in initial results: {doc_counts_total}")
        
        # Extract and return results
        evidence = []
        doc_counts = {}  # Track count of documents
        
        for match in results.matches:
            doc_name = match.metadata.get("document_name", "Unknown")
            
            # Skip if we already have max_duplicates_per_source entries from this document
            if doc_counts.get(doc_name, 0) >= max_duplicates_per_source:
                logger.info(f"Skipping entry from {doc_name} - already have enough from this source")
                continue
                
            # Determine content type and get appropriate text
            content_type = match.metadata.get("content_type", "text")
            
            # Get text based on content type
            if content_type == "image" or content_type == "image_description":
                text = match.metadata.get("description", "No description available")
                # Include caption if available
                caption = match.metadata.get("caption")
                if caption:
                    text = f"[Image Caption: {caption}]\n\n{text}"
                # Include image type if available
                image_type = match.metadata.get("image_type")
                if image_type:
                    text = f"[Image Type: {image_type}]\n\n{text}"
                
                # Evidence object for image content
                evidence_item = {
                    "score": match.score,
                    "document_name": doc_name,
                    "document_path": match.metadata.get("document_path", "Unknown"),
                    "page_number": match.metadata.get("page_number", -1),
                    "content_type": content_type,
                    "text": text,
                    "image_path": match.metadata.get("image_path", "")  # Include image path for images
                }
            else:
                text = match.metadata.get("text", "No text available")
                
                # Evidence object for text content
                evidence_item = {
                    "score": match.score,
                    "document_name": doc_name,
                    "document_path": match.metadata.get("document_path", "Unknown"),
                    "paragraph_index": match.metadata.get("paragraph_index", -1),
                    "content_type": content_type,
                    "text": text
                }
            
            evidence.append(evidence_item)
            
            # Increment the count for this document
            doc_counts[doc_name] = doc_counts.get(doc_name, 0) + 1
            
            # Stop once we have enough diverse evidence
            if len(evidence) >= top_k:
                break
        
        # Log final selection
        logger.info(f"Final evidence selection sources: {doc_counts}")
        return evidence
    except Exception as e:
        logger.error(f"Error searching for evidence: {e}")
        return []

@rate_limited_api_call
def generate_explanation(claim: str, evidence_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate an explanation of how the evidence supports the claim using Gemini"""
    if not evidence_list:
        return {"general_analysis": "No supporting evidence was found for this claim.", "evidence_assessments": []}
    
    try:
        # Prepare evidence texts for the prompt
        evidence_texts = []
        for i, evidence in enumerate(evidence_list):
            text = evidence.get("text", "")
            source = evidence.get("document_name", "Unknown source")
            content_type = evidence.get("content_type", "text")
            
            # Different location indicators based on content type
            if content_type == "image":
                location = f"page {evidence.get('page_number', 'unknown')}"
            else:
                location = f"paragraph index: {evidence.get('paragraph_index', -1)}"
                
            evidence_texts.append(f"Evidence {i+1} (from {source}, {location}, type: {content_type}):\n{text}")
        
        evidence_combined = "\n\n".join(evidence_texts)
        
        # Create prompt for Gemini
        prompt = f"""
        I need to analyze how the following evidence supports or refutes this claim:
        
        CLAIM: {claim}
        
        EVIDENCE:
        {evidence_combined}
        
        Important context about the evidence:
        - These text chunks were retrieved using semantic similarity to the claim
        - The retrieval process is based on embedding similarity, not literal text matching
        - Some chunks may be completely irrelevant or contain noise despite being retrieved
        - The paragraph index indicates which section of the document the text came from
        - Evidence labeled as "type: image" contains descriptions of visual content like tables, charts, or diagrams
        - For image evidence, the description was generated by AI analyzing the visual content
        
        Please provide a concise explanation of how the evidence relates to the claim. 
        Consider:
        1. Does the evidence directly support the claim?
        2. Does the evidence partially support the claim?
        3. Does the evidence contradict the claim?
        4. What specific aspects of the claim are addressed by the evidence?
        5. Are there any limitations or caveats in the evidence?
        
        Then, for EACH piece of evidence, provide:
        1. A relevancy score (1-5, where 5 is highly relevant and 1 is completely irrelevant)
           - Only assign high scores (4-5) when the evidence truly contains information relevant to the claim
           - Assign low scores (1-2) when the evidence is noise or unrelated despite appearing in search results
        2. An assessment of whether the evidence agrees, disagrees, partially agrees, or partially disagrees with the claim
           - If the evidence is irrelevant (score 1-2), mark it as "Not applicable"
        3. A brief reasoning explaining your assessment
        
        Format your response with a general analysis first, followed by the structured assessment of each evidence item.
        
        Example format:
        [General Analysis]
        
        [Evidence Assessments]
        Evidence 1:
        - Relevancy Score: X/5
        - Assessment: [Agrees/Disagrees/Partially Agrees/Partially Disagrees/Not applicable]
        - Reasoning: [Brief explanation]
        
        Evidence 2:
        - Relevancy Score: X/5
        - Assessment: [Agrees/Disagrees/Partially Agrees/Partially Disagrees/Not applicable]
        - Reasoning: [Brief explanation]
        """
        
        # Generate explanation using Gemini
        logger.info("Making API call to Gemini 1.5 Pro...")
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content(prompt)
        logger.info("Successfully received response from Gemini 1.5 Pro")
        
        # Extract the explanation
        explanation_text = response.text.strip()
        
        # Parse the response into structured format
        # First, split into general analysis and evidence assessments
        parts = explanation_text.split("[Evidence Assessments]", 1)
        
        general_analysis = parts[0].replace("[General Analysis]", "").strip()
        
        # Initialize evidence assessments list
        evidence_assessments = []
        
        # If we have evidence assessments section
        if len(parts) > 1:
            # Split by "Evidence X:" pattern
            assessment_texts = parts[1].strip().split("\nEvidence ")
            for assessment_text in assessment_texts:
                if not assessment_text.strip():
                    continue
                
                # If it doesn't start with a number (first split result), add "Evidence " back
                if not assessment_text.startswith("1:") and not assessment_text.startswith("2:") and not assessment_text.startswith("3:") and not assessment_text.startswith("4:") and not assessment_text.startswith("5:"):
                    assessment_text = "Evidence " + assessment_text
                
                # Extract evidence number
                evidence_num = 0
                if ":" in assessment_text.split("\n")[0]:
                    try:
                        evidence_num = int(assessment_text.split(":")[0].replace("Evidence ", "").strip())
                    except ValueError:
                        pass
                
                # Get paragraph index from the corresponding evidence item
                paragraph_index = -1
                if 0 < evidence_num <= len(evidence_list):
                    paragraph_index = evidence_list[evidence_num-1].get("paragraph_index", -1)
                
                # Extract relevancy score
                relevancy_score = 0
                relevancy_line = next((line for line in assessment_text.split("\n") if "Relevancy Score:" in line), "")
                if relevancy_line:
                    try:
                        relevancy_score = int(relevancy_line.split("Relevancy Score:")[1].split("/")[0].strip())
                    except (ValueError, IndexError):
                        pass
                
                # Extract assessment
                assessment = ""
                assessment_line = next((line for line in assessment_text.split("\n") if "Assessment:" in line), "")
                if assessment_line:
                    assessment = assessment_line.split("Assessment:")[1].strip()
                
                # Extract reasoning
                reasoning = ""
                reasoning_line = next((line for line in assessment_text.split("\n") if "Reasoning:" in line), "")
                if reasoning_line:
                    reasoning = reasoning_line.split("Reasoning:")[1].strip()
                
                # Add to evidence assessments
                if evidence_num > 0:
                    evidence_assessments.append({
                        "evidence_number": evidence_num,
                        "paragraph_index": paragraph_index,  # Include paragraph index
                        "relevancy_score": relevancy_score,
                        "assessment": assessment,
                        "reasoning": reasoning
                    })
        
        # Return structured result
        return {
            "general_analysis": general_analysis,
            "evidence_assessments": evidence_assessments,
            "raw_explanation": explanation_text  # Keep the raw text as well
        }
    
    except Exception as e:
        logger.error(f"Error generating explanation: {e}")
        return {
            "general_analysis": "Unable to generate explanation due to an error.",
            "evidence_assessments": [],
            "raw_explanation": f"Error: {str(e)}"
        }

def verify_claims(claims_file: str, output_file: str = None, top_k: int = 5, include_explanation: bool = True, ensure_source_diversity: bool = True, max_duplicates_per_source: int = 2):
    """Verify claims against the Pinecone index and save results"""
    # Load claims
    claims = load_claims(claims_file)
    logger.info(f"Loaded {len(claims)} claims from {claims_file}")
    
    # Process each claim
    results = []
    for claim in tqdm(claims, desc="Verifying claims"):
        claim_text = claim.get("claim", "")
        if not claim_text:
            continue
        
        # Preprocess the claim - use this for all operations except final storage
        preprocessed_claim = preprocess_claim(claim_text)
        
        # Search for evidence using preprocessed claim
        evidence = search_evidence(preprocessed_claim, top_k=top_k, max_duplicates_per_source=max_duplicates_per_source)
        
        # If ensuring source diversity and we have biased results, try again with tweaked query
        if ensure_source_diversity:
            # Check if all results are from same source
            source_counts = {}
            for e in evidence:
                doc_name = e.get("document_name", "Unknown")
                source_counts[doc_name] = source_counts.get(doc_name, 0) + 1
            
            # If we have strong source bias, retry with more neutral query
            if len(source_counts) <= 1 and evidence:
                logger.info(f"All evidence from same source. Trying more neutral query.")
                neutral_claim = f"In medicine, what is known about {claim_text}?"
                backup_evidence = search_evidence(neutral_claim, top_k=top_k, max_duplicates_per_source=max_duplicates_per_source)
                
                # Create set of backup sources for comparison
                backup_sources = set()
                for e in backup_evidence:
                    backup_sources.add(e.get("document_name", "Unknown"))
                
                # Merge results if we got different sources
                if len(backup_sources) > len(source_counts):
                    # Replace some results with backup evidence to increase diversity
                    evidence = evidence[:top_k//2] + backup_evidence[:top_k//2]
        
        # Generate explanation if requested - use preprocessed claim
        explanation = {}
        if include_explanation and evidence:
            # Log the preprocessed claim
            log_text = preprocessed_claim if len(preprocessed_claim) < 200 else f"{preprocessed_claim[:197]}..."
            logger.info(f"Generating explanation for claim: {log_text}")
            # Use preprocessed claim for explanation
            explanation = generate_explanation(preprocessed_claim, evidence)
        
        # Add to results - use original claim text, not the preprocessed one with context
        results.append({
            "claim": claim_text,  # Original claim without added context
            "evidence": evidence,
            "explanation": explanation
        })
    
    # Save results
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({"results": results}, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved results to {output_file}")
    
    return results

def format_custom_output(results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Format results in the custom JSON structure requested by the user"""
    formatted_claims = []
    
    for result in results:
        claim_text = result.get("claim", "")
        evidence_list = result.get("evidence", [])
        explanation = result.get("explanation", {})
        
        # Format evidence items
        formatted_evidence = []
        for evidence in evidence_list:
            content_type = evidence.get("content_type", "text")
            source = {
                "document_name": evidence.get("document_name", "Unknown"),
                "document_path": evidence.get("document_path", "Unknown"),
                "score": evidence.get("score", 0)
            }
            
            # Add additional metadata based on content type
            if content_type == "image" or content_type == "image_description":
                source["page_number"] = evidence.get("page_number", -1)
                source["image_path"] = evidence.get("image_path", "")  # Include image path
            else:
                source["paragraph_index"] = evidence.get("paragraph_index", -1)
            
            formatted_evidence.append({
                "text": evidence.get("text", ""),
                "content_type": content_type,
                "source": source
            })
        
        # Format and add the claim
        formatted_claim = {
            "claim": claim_text,
            "evidence": formatted_evidence
        }
        
        # Add explanation if available
        if explanation:
            formatted_claim["explanation"] = {
                "general_analysis": explanation.get("general_analysis", ""),
                "evidence_assessments": explanation.get("evidence_assessments", [])
            }
        
        formatted_claims.append(formatted_claim)
    
    return {"claims": formatted_claims}

def format_results(results: List[Dict[str, Any]], output_format: str = "md") -> str:
    """Format results in the specified format (markdown, html, or text)"""
    if output_format.lower() == "md":
        # Format as markdown
        output = "# Claims Verification Results\n\n"
        
        for i, result in enumerate(results):
            claim = result.get("claim", "")
            evidence = result.get("evidence", [])
            explanation = result.get("explanation", {})
            
            output += f"## Claim {i+1}\n\n"
            output += f"**{claim}**\n\n"
            
            # Add explanation if available
            if explanation:
                general_analysis = explanation.get("general_analysis", "")
                evidence_assessments = explanation.get("evidence_assessments", [])
                
                if general_analysis:
                    output += "### Analysis\n\n"
                    output += f"{general_analysis}\n\n"
                
                if evidence_assessments:
                    output += "### Evidence Assessments\n\n"
                    for assessment in evidence_assessments:
                        evidence_num = assessment.get("evidence_number", 0)
                        paragraph_index = assessment.get("paragraph_index", -1)
                        relevancy_score = assessment.get("relevancy_score", 0)
                        assessment_value = assessment.get("assessment", "")
                        reasoning = assessment.get("reasoning", "")
                        
                        output += f"#### Evidence {evidence_num}\n\n"
                        output += f"- **Relevancy Score**: {relevancy_score}/5\n"
                        output += f"- **Assessment**: {assessment_value}\n"
                        output += f"- **Reasoning**: {reasoning}\n\n"
            
            if evidence:
                output += "### Supporting Evidence\n\n"
                for j, item in enumerate(evidence):
                    score = item.get("score", 0)
                    doc_name = item.get("document_name", "Unknown")
                    content_type = item.get("content_type", "text")
                    
                    # Different location indicators based on content type
                    if content_type == "image" or content_type == "image_description":
                        location = f"Page: {item.get('page_number', 'unknown')}"
                        image_path = item.get("image_path", "")
                        if image_path:
                            location += f", Image Path: {image_path}"
                    else:
                        location = f"Paragraph: {item.get('paragraph_index', -1)}"
                        
                    text = item.get("text", "No text available")
                    
                    output += f"#### Evidence {j+1} (Score: {score:.4f}, Source: {doc_name}, {location}, Type: {content_type})\n\n"
                    output += f"{text}\n\n"
                    output += "---\n\n"
            else:
                output += "No supporting evidence found.\n\n"
                output += "---\n\n"
        
        return output
    
    else:  # Plain text
        # Format as plain text
        output = "CLAIMS VERIFICATION RESULTS\n"
        output += "=" * 40 + "\n\n"
        
        for i, result in enumerate(results):
            claim = result.get("claim", "")
            evidence = result.get("evidence", [])
            explanation = result.get("explanation", {})
            
            output += f"CLAIM {i+1}:\n"
            output += f"{claim}\n\n"
            
            # Add explanation if available
            if explanation:
                general_analysis = explanation.get("general_analysis", "")
                evidence_assessments = explanation.get("evidence_assessments", [])
                
                if general_analysis:
                    output += "ANALYSIS:\n"
                    output += "-" * 40 + "\n"
                    output += f"{general_analysis}\n\n"
                
                if evidence_assessments:
                    output += "EVIDENCE ASSESSMENTS:\n"
                    output += "-" * 40 + "\n"
                    for assessment in evidence_assessments:
                        evidence_num = assessment.get("evidence_number", 0)
                        paragraph_index = assessment.get("paragraph_index", -1)
                        relevancy_score = assessment.get("relevancy_score", 0)
                        assessment_value = assessment.get("assessment", "")
                        reasoning = assessment.get("reasoning", "")
                        
                        output += f"Evidence {evidence_num}:\n"
                        output += f"- Relevancy Score: {relevancy_score}/5\n"
                        output += f"- Assessment: {assessment_value}\n"
                        output += f"- Reasoning: {reasoning}\n\n"
            
            if evidence:
                output += "SUPPORTING EVIDENCE:\n\n"
                for j, item in enumerate(evidence):
                    score = item.get("score", 0)
                    doc_name = item.get("document_name", "Unknown")
                    content_type = item.get("content_type", "text")
                    
                    # Different location indicators based on content type
                    if content_type == "image" or content_type == "image_description":
                        location = f"Page: {item.get('page_number', 'unknown')}"
                        image_path = item.get("image_path", "")
                        if image_path:
                            location += f", Image Path: {image_path}"
                    else:
                        location = f"Paragraph: {item.get('paragraph_index', -1)}"
                        
                    text = item.get("text", "No text available")
                    
                    output += f"Evidence {j+1} (Score: {score:.4f}, Source: {doc_name}, {location}, Type: {content_type})\n"
                    output += "-" * 40 + "\n"
                    output += f"{text}\n\n"
            else:
                output += "No supporting evidence found.\n\n"
            
            output += "=" * 40 + "\n\n"
        
        return output

def main():
    """Main function to run the script"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Verify claims against clinical files using RAG")
    parser.add_argument("--claims_file", type=str, required=True,
                        help="JSON file containing claims to verify")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Output file to save results (JSON)")
    parser.add_argument("--custom_output_file", type=str, default=None,
                        help="Output file to save results in custom format (JSON)")
    parser.add_argument("--report_file", type=str, default=None,
                        help="Output file to save formatted report")
    parser.add_argument("--report_format", type=str, choices=["md", "txt"], default="md",
                        help="Format for the report (md, or txt)")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of evidence items to retrieve per claim")
    parser.add_argument("--no_explanation", action="store_true",
                        help="Skip generating explanations for claims")
    parser.add_argument("--no_source_diversity", action="store_true",
                        help="Disable source diversity checks")
    parser.add_argument("--max_duplicates_per_source", type=int, default=2,
                        help="Maximum number of evidence items allowed from the same source")
    args = parser.parse_args()
    
    # Verify claims
    results = verify_claims(
        claims_file=args.claims_file, 
        output_file=args.output_file, 
        top_k=args.top_k,
        include_explanation=not args.no_explanation,
        ensure_source_diversity=not args.no_source_diversity,
        max_duplicates_per_source=args.max_duplicates_per_source
    )
    
    # Generate and save custom format if requested
    if args.custom_output_file:
        custom_output = format_custom_output(results)
        with open(args.custom_output_file, 'w', encoding='utf-8') as f:
            json.dump(custom_output, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved custom format results to {args.custom_output_file}")
    
    # Generate and save report if requested
    if args.report_file:
        report = format_results(results, args.report_format)
        with open(args.report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Saved report to {args.report_file}")
    
    logger.info("Verification complete!")

if __name__ == "__main__":
    main() 