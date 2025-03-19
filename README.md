# Medical Claims Fact-Checking System

A RAG-based system for verifying medical claims against clinical documents using hybrid semantic search and AI analysis.

## Overview

This system allows you to:

1. Extract text and images from medical PDFs and create embeddings
2. Store these embeddings in a Pinecone vector database
3. Verify claims against the stored documents using hybrid semantic search
4. Generate AI-powered explanations of how evidence supports or refutes claims
5. Aggregate and analyze evidence to provide final verdicts
6. Output results in various formats (JSON, Markdown, HTML, or plain text)

## Prerequisites

- Python 3.8+
- [Poetry](https://python-poetry.org/docs/#installation) for dependency management
- Pinecone account (for vector database)
- Google AI API key (for Gemini embeddings and explanations)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/gremlin97/fact-check.git
   cd fact-check
   ```

2. Install dependencies using Poetry:
   ```bash
   poetry install
   ```

3. Create a `.env` file in the root directory with your API keys:
   ```
   GOOGLE_API_KEY=your_google_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   ```

## Usage

### 1. Processing PDF Documents

First, you need to process your PDF documents to extract text, images, and create embeddings:

```bash
poetry run python -m fact_check.process_all_pdfs
```

Options:
- `--start_index`: PDF index to start from (0-based)
- `--processed_file`: File to track processed PDFs for resuming later (default: "processed_pdfs.txt")
- `--skip_processed`: Skip PDFs that have already been processed
- `--verbose`: Enable more detailed logging

Example with options:
```bash
poetry run python -m fact_check.process_all_pdfs --start_index 0 --processed_file processed.txt --verbose
```

The processing pipeline includes:
1. Text extraction and chunking
2. Figure extraction using segmentation
3. Image analysis and description generation
4. Hybrid embedding generation (dense + sparse)
5. Vector storage in Pinecone

### 2. Verifying Claims

After processing your documents, you can verify claims against them:

```bash
poetry run python -m fact_check.verify_claims --claims_file claims.json --output_file results.json
```

Required arguments:
- `--claims_file`: JSON file containing claims to verify

Optional arguments:
- `--output_file`: Output file to save detailed results (JSON)
- `--custom_output_file`: Output file to save results in custom format (JSON)
- `--report_file`: Output file to save formatted report
- `--report_format`: Format for the report (md, html, or txt) (default: md)
- `--top_k`: Number of evidence items to retrieve per claim (default: 5)
- `--no_explanation`: Skip generating explanations for claims
- `--max_duplicates_per_source`: Maximum evidence items from same source (default: 2)
- `--ensure_source_diversity`: Whether to ensure evidence comes from diverse sources (default: true)

Example with all options:
```bash
poetry run python -m fact_check.verify_claims \
  --claims_file claims.json \
  --output_file detailed_results.json \
  --custom_output_file custom_results.json \
  --report_file report.md \
  --report_format md \
  --top_k 3 \
  --max_duplicates_per_source 2
```

### 3. Aggregating Evidence

After verifying claims, you can aggregate and analyze the evidence:

```bash
poetry run python -m fact_check.evidence_aggregator --input results.json --output aggregated_results.json
```

Options:
- `--input`: Input JSON file with fact-checking results
- `--output`: Output JSON file for aggregated results
- `--relevancy_threshold`: Minimum relevancy score (1-5) for evidence (default: 3.0)
- `--strong_support`: Threshold for 'Agree' verdict (default: 3.0)
- `--partial_support`: Threshold for 'Partially Agree' verdict (default: 0.0)
- `--use_evidence_score`: Use evidence retrieval score in weight calculation
- `--remove_duplicates`: Remove duplicate evidence items

### 4. Claims File Format

Your claims file should be in the following JSON format:

```json
{
  "claims": [
    {
      "claim": "Flublok ensures identical antigenic match with WHO- and FDA-selected flu strains."
    },
    {
      "claim": "Flublok contains 3x the hemagglutinin (HA) antigen content of standard-dose flu vaccines."
    }
  ]
}
```

### 5. Output Formats

#### Standard JSON Output

The standard JSON output includes detailed information about each claim, evidence, and explanation:

```json
{
  "results": [
    {
      "claim": "Flublok ensures identical antigenic match...",
      "evidence": [
        {
          "score": 0.8923,
          "document_name": "flublok_clinical_study.pdf",
          "document_path": "/path/to/flublok_clinical_study.pdf",
          "paragraph_index": 42,
          "content_type": "text",
          "text": "Flublok is manufactured using recombinant DNA technology..."
        },
        {
          "score": 0.8567,
          "document_name": "flublok_manufacturing.pdf",
          "document_path": "/path/to/flublok_manufacturing.pdf",
          "page_number": 15,
          "content_type": "image",
          "text": "Manufacturing process diagram showing recombinant DNA technology...",
          "image_path": "/path/to/image.png"
        }
      ],
      "explanation": {
        "general_analysis": "The evidence provides strong support for the claim...",
        "evidence_assessments": [
          {
            "relevancy_score": 5,
            "assessment": "Agrees",
            "reasoning": "Directly supports the claim about antigenic match..."
          }
        ]
      }
    }
  ]
}
```

#### Aggregated Results Output

The evidence aggregator provides a structured analysis of the evidence:

```json
{
  "aggregated_results": [
    {
      "claim": "Flublok ensures identical antigenic match...",
      "final_verdict": "Agree",
      "aggregated_score": 4.2,
      "evidence_breakdown": {
        "Agrees": {
          "count": 3,
          "total_weight": 12.5,
          "items": [1, 2, 4]
        },
        "Partially Agrees": {
          "count": 1,
          "total_weight": 2.1,
          "items": [3]
        }
      },
      "key_supporting_evidence": [
        {
          "evidence_number": 1,
          "document_name": "flublok_clinical_study.pdf",
          "paragraph_index": 42,
          "text": "Flublok is manufactured using recombinant DNA technology...",
          "assessment_type": "Agrees",
          "weight": 5.0,
          "reasoning": "Directly supports the claim..."
        }
      ]
    }
  ]
}
```

## Advanced Features

### Hybrid Search

The system uses a hybrid search approach combining:
- Dense embeddings (Gemini's embedding-001 model)
- Sparse embeddings (BM25 algorithm)

This provides better search results by combining the benefits of both semantic and keyword-based search.

### Image Processing

The system can:
1. Extract figures and diagrams from PDFs using segmentation
2. Generate detailed descriptions of visual content using Gemini Vision
3. Store image embeddings alongside text embeddings
4. Include visual evidence in claim verification

### Evidence Aggregation

The evidence aggregator provides:
1. Weighted scoring of evidence based on:
   - Relevancy scores (1-5)
   - Assessment types (Agrees/Disagrees/Partially Agrees)
   - Evidence retrieval scores (optional)
2. Source diversity tracking
3. Duplicate detection and handling
4. Final verdict determination
5. Detailed evidence breakdowns

### Rate Limiting

The system includes built-in rate limiting for API calls:
- Minimum 60-second delay between Gemini API calls
- Configurable batch sizes for Pinecone uploads
- Automatic retry logic for failed API calls

## Testing

You can test the PDF extraction without uploading to Pinecone:

```bash
poetry run python -m fact_check.extract --test --pdf_dir clinical_files
```

You can also test the embedding API before running a full verification:

```bash
poetry run python -m fact_check.verify_claims --claims_file claims.json --top_k 1
```

## Troubleshooting

### API Key Issues

If you encounter API key errors:
- Ensure your `.env` file is in the correct location
- Check that your API keys are valid and have the necessary permissions
- For testing without APIs, use the `--test` flag with the extract script

### Embedding Errors

If you see embedding-related errors:
- Ensure you're using the correct task type for embeddings
- Check your internet connection
- Verify your Google AI API key has access to the embedding models

### Pinecone Issues

If you have problems with Pinecone:
- Verify your Pinecone API key
- Check that you have permission to create indexes
- Ensure your Pinecone plan supports the dimensions and vector counts you're using

## Advanced Usage

### Custom Chunking

You can modify the `chunk_text_by_paragraphs` function in `extract.py` to customize how documents are split into chunks.

### Custom Embedding Models

To use different embedding models, modify the `get_embeddings` function in `extract.py` and the `get_embedding` function in `verify_claims.py`.

### Custom Explanation Prompts

To customize the AI explanations, modify the prompt in the `generate_explanation` function in `verify_claims.py`.

### Evidence Aggregation Configuration

You can customize the evidence aggregation process by modifying the configuration in `evidence_aggregator.py`:
- Adjust relevancy thresholds
- Modify weight calculations
- Change verdict thresholds
- Configure duplicate detection