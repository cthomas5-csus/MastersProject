"""
Fine-tune Legal-BERT for Transactional Clause Classification

This script trains Legal-BERT on the generated dataset and produces
experimental results for clause classification performance evaluation.
"""

import json
import torch
import logging
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClauseDataset(Dataset):
    """PyTorch Dataset for contract clauses"""

    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_dataset(dataset_path='dataset/legalbert_training_data.json'):
    """Load the generated dataset"""

    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    texts = [item['text'] for item in data]
    labels = [item['label'] for item in data]

    logger.info(f"Loaded {len(texts)} clauses from {dataset_path}")
    logger.info(f"  Transactional: {sum(labels)} ({sum(labels)/len(labels)*100:.1f}%)")
    logger.info(f"  Non-transactional: {len(labels) - sum(labels)} ({(len(labels)-sum(labels))/len(labels)*100:.1f}%)")

    return texts, labels

def train_model(
    train_loader,
    val_loader,
    model,
    tokenizer,
    optimizer,
    scheduler,
    device,
    epochs=3
):
    """Training loop"""

    best_val_acc = 0.0
    results = []

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")

        for batch in progress_bar:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=-1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)
        train_acc = accuracy_score(train_labels, train_preds)

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []

        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")

            for batch in progress_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                val_loss += outputs.loss.item()

                preds = torch.argmax(outputs.logits, dim=-1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / len(val_loader)

        # Calculate metrics
        val_acc = accuracy_score(val_labels, val_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            val_labels, val_preds, average='binary'
        )

        logger.info(f"\nEpoch {epoch+1}/{epochs} Results:")
        logger.info(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f}")
        logger.info(f"  Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        logger.info(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

        results.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_acc': train_acc,
            'val_loss': avg_val_loss,
            'val_acc': val_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path('models').mkdir(exist_ok=True)
            model.save_pretrained('models/legalbert_transactional_classifier')
            tokenizer.save_pretrained('models/legalbert_transactional_classifier')
            logger.info(f"  ✓ Saved best model and tokenizer (acc={val_acc:.4f})")

    return results

def evaluate_and_compare(model, test_loader, device, test_labels):
    """Evaluate final model and compare with rule-based baseline"""

    logger.info("\n" + "="*70)
    logger.info("FINAL EVALUATION - Legal-BERT vs Rule-Based Baseline")
    logger.info("="*70)

    # Legal-BERT evaluation
    model.eval()
    legalbert_preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating Legal-BERT"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=-1)
            legalbert_preds.extend(preds.cpu().numpy())

    # Calculate Legal-BERT metrics
    lb_accuracy = accuracy_score(test_labels, legalbert_preds)
    lb_precision, lb_recall, lb_f1, _ = precision_recall_fscore_support(
        test_labels, legalbert_preds, average='binary'
    )

    # Rule-based baseline (using existing classifier)
    from src.models.ml_clause_classifier import MLClauseClassifier
    from src.core.contract_processor import ContractClause

    # Load dataset to get texts
    with open('dataset/legalbert_training_data.json', 'r') as f:
        data = json.load(f)

    # Get test split texts (same split as used for training)
    _, test_texts, _, test_labels_split = train_test_split(
        [d['text'] for d in data],
        [d['label'] for d in data],
        test_size=0.2,
        random_state=42,
        stratify=[d['label'] for d in data]
    )

    # Run rule-based classifier
    rule_classifier = MLClauseClassifier()
    rule_preds = []

    for text in tqdm(test_texts, desc="Evaluating Rule-Based"):
        clause = ContractClause(id=0, text=text)
        classification = rule_classifier._classify_rule_based(clause)
        rule_preds.append(1 if classification.is_transactional else 0)

    # Calculate rule-based metrics
    rb_accuracy = accuracy_score(test_labels, rule_preds)
    rb_precision, rb_recall, rb_f1, _ = precision_recall_fscore_support(
        test_labels, rule_preds, average='binary'
    )

    # Print comparison table
    logger.info("\nLegal-BERT Classification Performance")
    logger.info("="*70)
    logger.info(f"{'Metric':<15} {'Legal-BERT':<15} {'Rule-Based':<15} {'Improvement':<15}")
    logger.info("-"*70)
    logger.info(f"{'Accuracy':<15} {lb_accuracy*100:>6.1f}%         {rb_accuracy*100:>6.1f}%         +{(lb_accuracy-rb_accuracy)*100:>5.1f}%")
    logger.info(f"{'Precision':<15} {lb_precision*100:>6.1f}%         {rb_precision*100:>6.1f}%         +{(lb_precision-rb_precision)*100:>5.1f}%")
    logger.info(f"{'Recall':<15} {lb_recall*100:>6.1f}%         {rb_recall*100:>6.1f}%         +{(lb_recall-rb_recall)*100:>5.1f}%")
    logger.info(f"{'F1-Score':<15} {lb_f1*100:>6.1f}%         {rb_f1*100:>6.1f}%         +{(lb_f1-rb_f1)*100:>5.1f}%")
    logger.info("="*70)

    # Save results
    results = {
        'legalbert': {
            'accuracy': float(lb_accuracy),
            'precision': float(lb_precision),
            'recall': float(lb_recall),
            'f1': float(lb_f1)
        },
        'rule_based': {
            'accuracy': float(rb_accuracy),
            'precision': float(rb_precision),
            'recall': float(rb_recall),
            'f1': float(rb_f1)
        },
        'improvement': {
            'accuracy': float(lb_accuracy - rb_accuracy),
            'precision': float(lb_precision - rb_precision),
            'recall': float(lb_recall - rb_recall),
            'f1': float(lb_f1 - rb_f1)
        }
    }

    with open('dataset/table6_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info("\nResults saved to: dataset/table6_results.json")

    return results

def main():
    """Main training pipeline"""

    logger.info("="*70)
    logger.info("LEGAL-BERT TRAINING FOR TRANSACTIONAL CLAUSE CLASSIFICATION")
    logger.info("="*70)

    # Load dataset
    texts, labels = load_dataset()

    # Split data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )

    logger.info(f"\nData split:")
    logger.info(f"  Train: {len(train_texts)} clauses")
    logger.info(f"  Val:   {len(val_texts)} clauses")
    logger.info(f"  Test:  {len(test_texts)} clauses")

    # Initialize model and tokenizer
    model_name = "nlpaueb/legal-bert-base-uncased"
    logger.info(f"\nLoading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    logger.info(f"Using device: {device}")

    # Create datasets
    train_dataset = ClauseDataset(train_texts, train_labels, tokenizer)
    val_dataset = ClauseDataset(val_texts, val_labels, tokenizer)
    test_dataset = ClauseDataset(test_texts, test_labels, tokenizer)

    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Optimizer and scheduler
    epochs = 3
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # Train
    logger.info(f"\nStarting training for {epochs} epochs...")
    logger.info("="*70)

    training_results = train_model(
        train_loader, val_loader, model, tokenizer, optimizer, scheduler, device, epochs
    )

    # Final evaluation
    logger.info("\n" + "="*70)
    logger.info("Training complete! Running final evaluation...")
    logger.info("="*70)

    # Load best model
    model = AutoModelForSequenceClassification.from_pretrained(
        'models/legalbert_transactional_classifier'
    )
    model.to(device)

    # Evaluate and compare
    final_results = evaluate_and_compare(model, test_loader, device, test_labels)

    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*70)
    logger.info("\nExperimental results have been generated!")
    logger.info("\nNext steps:")
    logger.info("  1. Review: dataset/table6_results.json")
    logger.info("  2. Use these metrics for performance evaluation")
    logger.info("  3. Update ml_clause_classifier.py to use trained model:")
    logger.info("     - Line 51: model_name = 'models/legalbert_transactional_classifier'")
    logger.info("     - Line 69: self.use_fallback = False")
    logger.info("="*70)

if __name__ == "__main__":
    main()
