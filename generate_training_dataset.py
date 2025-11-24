"""
Generate Synthetic Contract Clause Dataset for Legal-BERT Training

Uses Claude API to generate realistic contract clauses with labels.
This creates a proper training dataset for fine-tuning Legal-BERT.
"""

import json
import os
import time
import logging
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_clause_batch(client, batch_num, clauses_per_batch=20):
    """Generate a batch of labeled contract clauses"""

    prompt = f"""Generate exactly {clauses_per_batch} realistic legal contract clauses in JSON format.

You must create 10 TRANSACTIONAL clauses (containing specific obligations, payments, deliverables, deadlines)
and 10 NON-TRANSACTIONAL clauses (definitions, recitals, boilerplate, governing law, etc.).

TRANSACTIONAL EXAMPLES:
- Payment obligations with specific amounts and deadlines
- Delivery requirements with dates and specifications
- Performance milestones with measurable criteria
- Penalty clauses with specific consequences
- Service level agreements with metrics

NON-TRANSACTIONAL EXAMPLES:
- Legal definitions and interpretations
- Whereas clauses and recitals
- Governing law and jurisdiction
- Severability and general provisions
- Signature blocks and witnessing clauses

Format your response as a JSON array:

[
  {{
    "text": "The Buyer shall pay the Seller $50,000 within 30 days of delivery of goods to the specified warehouse location.",
    "label": 1,
    "category": "payment_obligation"
  }},
  {{
    "text": "This Agreement shall be governed by and construed in accordance with the laws of the State of California, without regard to its conflict of law provisions.",
    "label": 0,
    "category": "governing_law"
  }},
  ...
]

Generate {clauses_per_batch} diverse, realistic clauses now (10 transactional with label=1, 10 non-transactional with label=0):"""

    try:
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=4000,
            temperature=0.8,  # Higher for diversity
            messages=[{"role": "user", "content": prompt}]
        )

        # Extract JSON from response
        response_text = response.content[0].text

        # Find JSON array in response
        start_idx = response_text.find('[')
        end_idx = response_text.rfind(']') + 1

        if start_idx == -1 or end_idx == 0:
            logger.error(f"No JSON found in response for batch {batch_num}")
            return []

        json_text = response_text[start_idx:end_idx]
        clauses = json.loads(json_text)

        logger.info(f"Batch {batch_num}: Generated {len(clauses)} clauses")

        # Validate labels
        transactional_count = sum(1 for c in clauses if c.get('label') == 1)
        non_transactional_count = sum(1 for c in clauses if c.get('label') == 0)

        logger.info(f"  Transactional: {transactional_count}, Non-transactional: {non_transactional_count}")

        return clauses

    except Exception as e:
        logger.error(f"Error generating batch {batch_num}: {e}")
        return []

def generate_dataset(target_size=300, batch_size=20):
    """Generate complete dataset of labeled clauses"""

    # Check for API key
    api_key = os.getenv('CLAUDE_API_KEY')
    if not api_key:
        raise ValueError("CLAUDE_API_KEY environment variable not set")

    client = Anthropic(api_key=api_key)

    all_clauses = []
    num_batches = (target_size + batch_size - 1) // batch_size

    # Create dataset directory
    Path('dataset').mkdir(exist_ok=True)
    output_file = 'dataset/legalbert_training_data.json'

    logger.info(f"Generating {target_size} clauses in {num_batches} batches")
    logger.info("="*70)

    for batch_num in range(1, num_batches + 1):
        logger.info(f"\nGenerating batch {batch_num}/{num_batches}...")

        clauses = generate_clause_batch(client, batch_num, batch_size)

        # Add IDs
        for clause in clauses:
            clause['id'] = len(all_clauses) + 1
            all_clauses.append(clause)

        # Save incrementally after each batch
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_clauses, f, indent=2, ensure_ascii=False)

        logger.info(f"  Saved {len(all_clauses)} clauses so far to {output_file}")

        # Rate limiting
        if batch_num < num_batches:
            logger.info("Waiting 2 seconds before next batch...")
            time.sleep(2)

    logger.info("\n" + "="*70)
    logger.info("DATASET GENERATION COMPLETE")
    logger.info("="*70)

    # Statistics
    total = len(all_clauses)
    transactional = sum(1 for c in all_clauses if c.get('label') == 1)
    non_transactional = sum(1 for c in all_clauses if c.get('label') == 0)

    logger.info(f"\nTotal clauses: {total}")
    if total > 0:
        logger.info(f"Transactional: {transactional} ({transactional/total*100:.1f}%)")
        logger.info(f"Non-transactional: {non_transactional} ({non_transactional/total*100:.1f}%)")
    else:
        logger.error("No clauses were generated! Check API key and model availability.")

    # Save dataset
    Path('dataset').mkdir(exist_ok=True)
    output_file = 'dataset/legalbert_training_data.json'

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_clauses, f, indent=2, ensure_ascii=False)

    logger.info(f"\nDataset saved to: {output_file}")

    return all_clauses

if __name__ == "__main__":
    print("="*70)
    print("LEGAL-BERT TRAINING DATASET GENERATOR")
    print("="*70)
    print("\nThis will use Claude API to generate realistic contract clauses")
    print("with labels for training Legal-BERT.\n")

    # Check API key
    if not os.getenv('CLAUDE_API_KEY'):
        print("ERROR: CLAUDE_API_KEY environment variable not set")
        print("\nSet it with:")
        print("  export CLAUDE_API_KEY='your-api-key-here'")
        exit(1)

    # Generate dataset
    target_size = 600  # Larger dataset for more robust metrics
    print(f"Target dataset size: {target_size} clauses")
    print("="*70 + "\n")

    dataset = generate_dataset(target_size=target_size)

    print("\n" + "="*70)
    print("SUCCESS!")
    print("="*70)
    print(f"\nGenerated {len(dataset)} labeled clauses")
    print("\nNext steps:")
    print("  1. Review: dataset/legalbert_training_data.json")
    print("  2. Run: python train_legalbert.py")
    print("  3. Evaluate and analyze classification metrics")
    print("="*70)
