import json
import logging
import argparse
from typing import List, Dict, Any, Tuple, Optional
import os
import re
from collections import defaultdict

# Configure logging
log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evidence_aggregator.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Default configuration parameters
DEFAULT_CONFIG = {
    "relevancy_score_threshold": 3,  # Minimum relevancy score (1-5 scale)
    "base_weights": {
        "Agrees": 1.0,
        "Partially Agrees": 0.5,
        "Disagrees": -1.0,
        "Not applicable": 0.0
    },
    "final_verdict_thresholds": {
        "strong_support": 3.0,  # Above this = "Agree"
        "partial_support": 0.0   # Between 0 and strong_support = "Partially Agree", below 0 = "Disagree"
    },
    "duplicate_similarity_threshold": 0.9,  # 90% text similarity threshold
    "use_evidence_score": False,  # Whether to factor in the retrieval score
    "remove_duplicates": False    # Whether to remove duplicate evidence (set to false as requested)
}

class EvidenceAggregator:
    """
    Processes fact-checking outputs to determine if claims are supported by evidence
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize with configuration parameters"""
        self.config = DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        logger.info(f"Initialized EvidenceAggregator with config: {self.config}")
    
    def process_file(self, input_file: str, output_file: str = None) -> Dict[str, Any]:
        """Process an input file and return aggregated results"""
        logger.info(f"Processing input file: {input_file}")
        
        # Validate input file exists
        if not os.path.exists(input_file):
            error_msg = f"Input file not found: {input_file}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Validate file is of proper type
        if not input_file.endswith('.json'):
            logger.warning(f"Input file {input_file} may not be a JSON file. Attempting to process anyway.")
        
        try:
            # Load and validate input data
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Validate basic structure
            if not isinstance(data, dict) or "results" not in data:
                error_msg = f"Invalid input format: Expected top-level 'results' key"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Process each claim
            results = []
            for claim_data in data["results"]:
                try:
                    result = self.process_claim(claim_data)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing claim: {e}")
                    # Continue with other claims
            
            # Prepare output
            output_data = {"aggregated_results": results}
            
            # Save to output file if specified
            if output_file:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                logger.info(f"Results saved to {output_file}")
            
            return output_data
            
        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse JSON from {input_file}: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            logger.error(f"Unexpected error processing file {input_file}: {str(e)}")
            raise
    
    def process_claim(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single claim and its evidence"""
        try:
            claim_text = claim_data.get("claim", "")
            evidence_list = claim_data.get("evidence", [])
            explanation = claim_data.get("explanation", {})
            
            logger.info(f"Processing claim: {claim_text[:100]}...")
            
            # Validate required fields
            self._validate_claim_data(claim_data)
            
            # Apply filters
            filtered_evidence = self._apply_filters(evidence_list, explanation)
            logger.info(f"After filtering: {len(filtered_evidence)} evidence items remain")
            
            # Apply weights and aggregate
            weighted_evidence, aggregate_score = self._apply_weights_and_aggregate(filtered_evidence, explanation)
            
            # Determine verdict
            verdict = self._determine_verdict(aggregate_score)
            
            # Get evidence category breakdown
            breakdown = self._get_category_breakdown(weighted_evidence)
            
            # Get key supporting evidence
            key_evidence = self._get_key_evidence(weighted_evidence)
            
            # Create result object
            result = {
                "claim": claim_text,
                "final_verdict": verdict,
                "aggregated_score": aggregate_score,
                "evidence_breakdown": breakdown,
                "key_supporting_evidence": key_evidence,
                "filtering_log": self.filtering_log,
                "configuration": self.config
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in process_claim: {e}")
            raise
    
    def _validate_claim_data(self, claim_data: Dict[str, Any]) -> None:
        """Validate the structure and fields of claim data"""
        # Check for required top-level keys
        required_keys = ["claim", "evidence", "explanation"]
        for key in required_keys:
            if key not in claim_data:
                error_msg = f"Missing required key in claim data: {key}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        # Validate evidence items
        evidence_list = claim_data["evidence"]
        if not isinstance(evidence_list, list):
            error_msg = "Evidence must be a list"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        for i, evidence in enumerate(evidence_list):
            try:
                # Check required fields
                required_evidence_fields = ["score", "document_name", "paragraph_index", "text"]
                for field in required_evidence_fields:
                    if field not in evidence:
                        logger.warning(f"Evidence item {i} missing field: {field}")
                
                # Validate data types
                if "score" in evidence and not isinstance(evidence["score"], (int, float)):
                    logger.warning(f"Evidence item {i}: score is not numeric")
                
                if "paragraph_index" in evidence and not isinstance(evidence["paragraph_index"], (int, float)):
                    logger.warning(f"Evidence item {i}: paragraph_index is not numeric")
            
            except Exception as e:
                logger.error(f"Error validating evidence item {i}: {e}")
        
        # Validate explanation
        explanation = claim_data["explanation"]
        if not isinstance(explanation, dict):
            error_msg = "Explanation must be a dictionary"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Check for evidence assessments
        if "evidence_assessments" in explanation:
            assessments = explanation["evidence_assessments"]
            if not isinstance(assessments, list):
                logger.warning("evidence_assessments is not a list")
            
            for i, assessment in enumerate(assessments):
                try:
                    # Check required fields
                    required_assessment_fields = ["evidence_number", "relevancy_score", "assessment", "reasoning"]
                    for field in required_assessment_fields:
                        if field not in assessment:
                            logger.warning(f"Assessment {i} missing field: {field}")
                    
                    # Validate data types
                    if "relevancy_score" in assessment and not isinstance(assessment["relevancy_score"], (int, float)):
                        logger.warning(f"Assessment {i}: relevancy_score is not numeric")
                    
                    if "assessment" in assessment:
                        valid_assessments = ["Agrees", "Partially Agrees", "Disagrees", "Not applicable"]
                        if assessment["assessment"] not in valid_assessments:
                            logger.warning(f"Assessment {i} has invalid value: {assessment['assessment']}")
                
                except Exception as e:
                    logger.error(f"Error validating assessment {i}: {e}")
    
    def _apply_filters(self, evidence_list: List[Dict[str, Any]], explanation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply all filtering logic to evidence items"""
        # Initialize filtering log
        self.filtering_log = []
        
        # Create a mapping from evidence_number to evidence item
        evidence_map = {i+1: item for i, item in enumerate(evidence_list)}
        
        # Get assessments
        assessments = explanation.get("evidence_assessments", [])
        
        # If no assessments are available, create synthetic assessments based on evidence scores
        if not assessments:
            logger.warning("No evidence assessments found. Creating synthetic assessments based on evidence scores.")
            assessments = []
            for i, evidence in enumerate(evidence_list):
                score = evidence.get("score", 0)
                # Convert score (typically 0-1) to relevancy score (1-5)
                # Assuming scores > 0.7 are highly relevant (4-5), 0.5-0.7 are moderately relevant (3),
                # and < 0.5 are less relevant (1-2)
                if score > 0.8:
                    relevancy_score = 5
                    assessment_type = "Agrees"
                elif score > 0.7:
                    relevancy_score = 4
                    assessment_type = "Agrees"
                elif score > 0.6:
                    relevancy_score = 3
                    assessment_type = "Partially Agrees"
                elif score > 0.5:
                    relevancy_score = 2
                    assessment_type = "Partially Agrees"
                else:
                    relevancy_score = 1
                    assessment_type = "Not applicable"
                
                assessments.append({
                    "evidence_number": i + 1,
                    "paragraph_index": evidence.get("paragraph_index", -1),
                    "relevancy_score": relevancy_score,
                    "assessment": assessment_type,
                    "reasoning": f"Synthetic assessment based on evidence score {score:.2f}"
                })
            
            # Add synthetic assessments to explanation for later use
            explanation["evidence_assessments"] = assessments
            logger.info(f"Created {len(assessments)} synthetic assessments from evidence scores")
        
        # Track which evidence items to keep
        keep_evidence = set(range(1, len(evidence_list) + 1))
        
        # Filter 1: Relevancy Score Filter
        threshold = self.config["relevancy_score_threshold"]
        for assessment in assessments:
            evidence_num = assessment.get("evidence_number", 0)
            relevancy_score = assessment.get("relevancy_score", 0)
            
            if evidence_num in keep_evidence and relevancy_score < threshold:
                keep_evidence.remove(evidence_num)
                self.filtering_log.append({
                    "evidence_number": evidence_num,
                    "filter": "Relevancy Score Filter",
                    "reason": f"Score {relevancy_score} below threshold {threshold}"
                })
                logger.info(f"Filter 1: Removed evidence {evidence_num} due to low relevancy score {relevancy_score}")
        
        # Filter 2: Assessment Type Filter
        for assessment in assessments:
            evidence_num = assessment.get("evidence_number", 0)
            assessment_type = assessment.get("assessment", "")
            
            if evidence_num in keep_evidence and assessment_type == "Not applicable":
                keep_evidence.remove(evidence_num)
                self.filtering_log.append({
                    "evidence_number": evidence_num,
                    "filter": "Assessment Type Filter",
                    "reason": "Assessment is 'Not applicable'"
                })
                logger.info(f"Filter 2: Removed evidence {evidence_num} due to 'Not applicable' assessment")
        
        # Get filtered evidence list based on current keep_evidence set
        filtered_evidence = [evidence_map[num] for num in keep_evidence if num in evidence_map]
        
        # Filter 3: Duplicate Evidence Removal (conditional based on config)
        if self.config.get("remove_duplicates", False):
            filtered_evidence, duplicate_log = self._remove_duplicates(filtered_evidence)
            self.filtering_log.extend(duplicate_log)
            logger.info(f"Filter 3: Applied duplicate removal, {len(duplicate_log)} duplicates found")
        else:
            # Add a note to the filtering log that we're keeping duplicates
            self.filtering_log.append({
                "filter": "Duplicate Evidence",
                "reason": "Duplicate evidence items are being kept as requested"
            })
            logger.info("Filter 3: Skipping duplicate removal as configured")
        
        # Filter 4: Content and Numerical Matching (optional)
        filtered_evidence, content_log = self._filter_by_content_match(filtered_evidence, explanation, assessments)
        self.filtering_log.extend(content_log)
        
        return filtered_evidence
    
    def _filter_by_content_match(self, evidence_list: List[Dict[str, Any]], 
                               explanation: Dict[str, Any], 
                               assessments: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Optional filter for checking content matches in quantitative claims
        Returns filtered evidence list and logging entries
        """
        # This is a placeholder for the optional numerical match filter
        # In a full implementation, you would:
        # 1. Extract numerical values from the claim
        # 2. Check if these values are mentioned in the evidence
        # 3. Flag evidence that doesn't mention key numbers
        
        # For now, we'll just return the evidence list unchanged
        return evidence_list, []
    
    def _remove_duplicates(self, evidence_list: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Remove duplicate evidence items based on text similarity
        Returns filtered evidence list and filtering log entries
        """
        # Early return if no duplicates are possible
        if len(evidence_list) <= 1:
            return evidence_list, []
        
        # Get threshold from config
        threshold = self.config.get("duplicate_similarity_threshold", 0.9)
        
        # Prepare output variables
        filtered_evidence = []
        filtering_log = []
        seen_texts = []
        seen_ids = set()
        
        # Helper function to check text similarity
        def is_duplicate(text1, text2, threshold):
            # Simple character-level similarity
            if not text1 or not text2:
                return False
            
            # Calculate overlap
            shorter = text1 if len(text1) <= len(text2) else text2
            longer = text2 if len(text1) <= len(text2) else text1
            
            # If the shorter text is empty, they're not duplicates
            if not shorter:
                return False
            
            # Simple overlap calculation
            overlap = sum(1 for c in shorter if c in longer) / len(shorter)
            return overlap >= threshold
        
        # Process each evidence item
        for item in evidence_list:
            item_id = item.get("evidence_number", 0)
            text = item.get("text", "").strip()
            
            # Check if this item is a duplicate of any we've seen
            is_dup = False
            for seen_text in seen_texts:
                if is_duplicate(text, seen_text, threshold):
                    is_dup = True
                    break
            
            if is_dup:
                # Log that we're filtering out this duplicate
                filtering_log.append({
                    "evidence_number": item_id,
                    "filter": "Duplicate Content",
                    "reason": f"Similar content to another evidence item (threshold: {threshold})"
                })
                logger.info(f"Removed duplicate evidence {item_id}")
            else:
                # Not a duplicate, keep it
                filtered_evidence.append(item)
                seen_texts.append(text)
                seen_ids.add(item_id)
        
        return filtered_evidence, filtering_log
    
    def _apply_weights_and_aggregate(self, evidence_list: List[Dict[str, Any]], 
                                    explanation: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], float]:
        """Apply weights to evidence items and aggregate the score"""
        # Get assessments
        assessments = explanation.get("evidence_assessments", [])
        
        # Process each evidence item directly with its synthetic assessment
        weighted_evidence = []
        total_score = 0.0
        
        # Process the evidence items with their corresponding assessments
        for i, evidence in enumerate(evidence_list):
            # Use index-based matching instead of paragraph_index
            if i < len(assessments):
                assessment = assessments[i]
                
                assessment_type = assessment.get("assessment", "")
                relevancy_score = assessment.get("relevancy_score", 0)
                
                # Apply base weight by assessment type
                base_weight = self.config["base_weights"].get(assessment_type, 0.0)
                
                # Multiply by relevancy score
                weight = base_weight * relevancy_score
                
                # Optionally multiply by evidence score
                if self.config["use_evidence_score"]:
                    evidence_score = evidence.get("score", 1.0)
                    weight *= evidence_score
                
                # Start with the base weighted item
                weighted_item = {
                    "evidence_number": i + 1,
                    "document_name": evidence.get("document_name", ""),
                    "document_path": evidence.get("document_path", ""),
                    "paragraph_index": evidence.get("paragraph_index", -1),
                    "text": evidence.get("text", ""),
                    "assessment_type": assessment_type,
                    "relevancy_score": relevancy_score,
                    "base_weight": base_weight,
                    "final_weight": weight,
                    "reasoning": assessment.get("reasoning", "")
                }
                
                # Preserve important fields from the evidence item
                for field in ["image_path", "content_type", "page_number", "source"]:
                    if field in evidence:
                        weighted_item[field] = evidence[field]
                
                # Log if we found image-related fields
                if "image_path" in evidence or "content_type" in evidence and "image" in str(evidence.get("content_type")).lower():
                    logger.info(f"Evidence {i+1} has image data: path={evidence.get('image_path')}, type={evidence.get('content_type')}")
                
                weighted_evidence.append(weighted_item)
                total_score += weight
                
                logger.info(f"Evidence {i+1}: {assessment_type} with relevancy {relevancy_score} → weight {weight}")
        
        return weighted_evidence, total_score
    
    def _determine_verdict(self, aggregate_score: float) -> str:
        """Determine the final verdict based on the aggregate score"""
        thresholds = self.config["final_verdict_thresholds"]
        
        if aggregate_score >= thresholds["strong_support"]:
            verdict = "Agree"
        elif aggregate_score >= thresholds["partial_support"]:
            verdict = "Partially Agree"
        else:
            verdict = "Disagree/Not Supported"
        
        logger.info(f"Final verdict: {verdict} (Score: {aggregate_score})")
        return verdict
    
    def _get_category_breakdown(self, weighted_evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get a breakdown of evidence by category"""
        categories = {
            "Agrees": {"count": 0, "total_weight": 0.0, "items": []},
            "Partially Agrees": {"count": 0, "total_weight": 0.0, "items": []},
            "Disagrees": {"count": 0, "total_weight": 0.0, "items": []},
            "Other": {"count": 0, "total_weight": 0.0, "items": []},  # Fallback category
        }
        
        for item in weighted_evidence:
            assessment_type = item.get("assessment_type", "Other")
            
            if assessment_type in categories:
                categories[assessment_type]["count"] += 1
                categories[assessment_type]["total_weight"] += item.get("final_weight", 0.0)
                categories[assessment_type]["items"].append(item.get("evidence_number", 0))
            else:
                # If we encounter an unexpected assessment type
                categories["Other"]["count"] += 1
                categories["Other"]["total_weight"] += item.get("final_weight", 0.0)
                categories["Other"]["items"].append(item.get("evidence_number", 0))
                logger.warning(f"Unexpected assessment type: {assessment_type}")
        
        # Remove "Other" category if empty
        if categories["Other"]["count"] == 0:
            del categories["Other"]
        
        return categories
    
    def _get_key_evidence(self, weighted_evidence: List[Dict[str, Any]], top_n: int = 3) -> List[Dict[str, Any]]:
        """Get the most influential evidence items"""
        # Sort by absolute weight value (to get both strong positive and negative)
        sorted_evidence = sorted(weighted_evidence, key=lambda x: abs(x["final_weight"]), reverse=True)
        
        # Take the top N items
        key_evidence = []
        for item in sorted_evidence[:top_n]:
            # Create a copy of the important fields
            evidence_item = {
                "evidence_number": item["evidence_number"],
                "document_name": item["document_name"],
                "paragraph_index": item["paragraph_index"],
                "text": item["text"],
                "assessment_type": item["assessment_type"],
                "weight": item["final_weight"],
                "reasoning": item["reasoning"]
            }
            
            # Preserve image-related fields if they exist
            for field in ["image_path", "content_type", "page_number"]:
                if field in item:
                    evidence_item[field] = item[field]
            
            # Add any other metadata that might be useful for display
            if "source" in item:
                evidence_item["source"] = item["source"]
            
            key_evidence.append(evidence_item)
        
        return key_evidence


def main():
    """Main function to run the script"""
    parser = argparse.ArgumentParser(description="Aggregate evidence from fact-checking output")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input JSON file with fact-checking results")
    parser.add_argument("--output", "-o", type=str, default=None, help="Output JSON file for aggregated results")
    parser.add_argument("--relevancy_threshold", "-r", type=float, default=3.0, 
                        help="Minimum relevancy score (1-5) for evidence to be considered")
    parser.add_argument("--strong_support", "-s", type=float, default=3.0,
                        help="Threshold for 'Agree' verdict (aggregate score)")
    parser.add_argument("--partial_support", "-p", type=float, default=0.0,
                        help="Threshold for 'Partially Agree' verdict (aggregate score)")
    parser.add_argument("--use_evidence_score", "-e", action="store_true",
                        help="Use evidence retrieval score in weight calculation")
    parser.add_argument("--remove_duplicates", "-d", action="store_true",
                        help="Remove duplicate evidence items (default is to keep duplicates)")
    
    args = parser.parse_args()
    
    # Create custom config from arguments
    config = {
        "relevancy_score_threshold": args.relevancy_threshold,
        "final_verdict_thresholds": {
            "strong_support": args.strong_support,
            "partial_support": args.partial_support
        },
        "use_evidence_score": args.use_evidence_score,
        "remove_duplicates": args.remove_duplicates
    }
    
    # If no output file provided, create one based on input filename
    if not args.output:
        base_name = os.path.splitext(args.input)[0]
        args.output = f"{base_name}_aggregated.json"
    
    try:
        # Initialize and run the aggregator
        aggregator = EvidenceAggregator(config)
        results = aggregator.process_file(args.input, args.output)
        
        print(f"\n{'=' * 40}")
        print(f"Processing complete. Results saved to {args.output}")
        print(f"{'=' * 40}\n")
        
        # Display summary of results
        print("SUMMARY OF RESULTS:")
        print(f"{'-' * 40}")
        for i, result in enumerate(results["aggregated_results"]):
            claim_text = result.get('claim', '')
            display_claim = (claim_text[:97] + '...') if len(claim_text) > 100 else claim_text
            
            print(f"\nClaim {i+1}: {display_claim}")
            print(f"Verdict: {result['final_verdict']} (Score: {result['aggregated_score']:.2f})")
            
            # Format the evidence breakdown
            agrees_count = result['evidence_breakdown'].get('Agrees', {}).get('count', 0)
            partially_count = result['evidence_breakdown'].get('Partially Agrees', {}).get('count', 0)
            disagrees_count = result['evidence_breakdown'].get('Disagrees', {}).get('count', 0)
            
            print(f"Evidence breakdown: {agrees_count} agree, {partially_count} partially agree, {disagrees_count} disagree")
            print(f"{'-' * 40}")
    
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        logger.error(f"Error in main function: {e}", exc_info=True)
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code) 