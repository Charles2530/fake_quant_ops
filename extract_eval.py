#!/usr/bin/env python3
"""
Extract evaluation metrics at step 3000 from CSV files and generate a table.
Usage: python extract_eval.py [eval_folder_path]
Example: python extract_eval.py logs/OLMo-7b/eval
"""

import csv
import os
import sys
import glob
from collections import defaultdict

# Model-specific configurations
MODEL_CONFIGS = {
    'OLMo-7B': {
        'target_step': 3000,
        'model_name_mapping': {
            'OLMo-7B-reproduce_20251216_220018': 'BF16',
            'FakeQuant-Activation-OLMo-7B-MXFP-4-auto-reverse_20251226_215924': 'Four Over Six',
            'FakeQuant-Activation-OLMo-7B-MXFP-4-Minus-auto_20251224_091551': 'TRACE',
            'FakeQuant-Activation-OLMo-7B-MXFP-4-Minus1_20251221_012126': 'Fixed Exp-1',
            'FakeQuant-Activation-OLMo-7B-MXFP-4-Minus2_20251221_011812': 'Fixed Exp-2',
            'FakeQuant-Activation-OLMo-7B-MXFP-4-Minus-auto-2_20251226_181119': 'Mixed',
        },
        'row_order': ['BF16', 'mxfp4', 'Four Over Six', 'Fixed Exp-1', 'Fixed Exp-2', 'Mixed', 'TRACE'],
        'train_loss_file_mapping': {
            'OLMo-7B-reproduce': 'OLMo-7B-reproduce_20251216_220018',
            'FakeQuant-Activation-OLMo-7B-MXFP-4-auto-reverse': 'FakeQuant-Activation-OLMo-7B-MXFP-4-auto-reverse_20251226_215924',
            'FakeQuant-Activation-OLMo-7B-MXFP-4-Minus-auto': 'FakeQuant-Activation-OLMo-7B-MXFP-4-Minus-auto_20251224_091551',
            'FakeQuant-Activation-OLMo-7B-MXFP-4-Minus1': 'FakeQuant-Activation-OLMo-7B-MXFP-4-Minus1_20251221_012126',
            'FakeQuant-Activation-OLMo-7B-MXFP-4-Minus2': 'FakeQuant-Activation-OLMo-7B-MXFP-4-Minus2_20251221_011812',
            'FakeQuant-Activation-OLMo-7B-MXFP-4-Minus-auto-2': 'FakeQuant-Activation-OLMo-7B-MXFP-4-Minus-auto-2_20251226_181119',
        },
        'train_loss_model_to_table': {
            'OLMo-7B-reproduce_20251216_220018': 'BF16',
            'FakeQuant-Activation-OLMo-7B-MXFP-4-auto-reverse_20251226_215924': 'Four Over Six',
            'FakeQuant-Activation-OLMo-7B-MXFP-4-Minus-auto_20251224_091551': 'TRACE',
            'FakeQuant-Activation-OLMo-7B-MXFP-4-Minus1_20251221_012126': 'Fixed Exp-1',
            'FakeQuant-Activation-OLMo-7B-MXFP-4-Minus2_20251221_011812': 'Fixed Exp-2',
            'FakeQuant-Activation-OLMo-7B-MXFP-4-Minus-auto-2_20251226_181119': 'Mixed',
        },
        'csv_subfolder': '',  # CSV files are in logs/ directory
    },
    'OLMo-1B': {
        'target_step': 2000,
        'model_name_mapping': {
            'OLMo-1B-reproduce_20251223_125914': 'BF16',
            'FakeQuant-Activation-OLMo-1B-MXFP-4_20251227_235622': 'mxfp4 (Linear)',
            'FakeQuant-Activation-OLMo-1B-MXFP-4-attn_20251225_121524': 'mxfp4 (A+L)',
        },
        'row_order': ['BF16', 'mxfp4 (Linear)', 'mxfp4 (A+L)', 'TRACE (Linear)', 'TRACE (A+L)'],
        'train_loss_file_mapping': {
            'OLMo-1B-reproduce': 'OLMo-1B-reproduce_20251223_125914',
            'FakeQuant-Activation-OLMo-1B-MXFP-4': 'FakeQuant-Activation-OLMo-1B-MXFP-4_20251227_235622',
            'FakeQuant-Activation-OLMo-1B-MXFP-4-attn': 'FakeQuant-Activation-OLMo-1B-MXFP-4-attn_20251225_121524',
        },
        'train_loss_model_to_table': {
            'OLMo-1B-reproduce_20251223_125914': 'BF16',
            'FakeQuant-Activation-OLMo-1B-MXFP-4_20251227_235622': 'mxfp4 (Linear)',
            'FakeQuant-Activation-OLMo-1B-MXFP-4-attn_20251225_121524': 'mxfp4 (A+L)',
        },
        'csv_subfolder': 'csv',  # CSV files are in logs/OLMo-1b/csv/ directory
    },
}

# Legacy mappings for backward compatibility (deprecated, use MODEL_CONFIGS instead)
MODEL_NAME_MAPPING = MODEL_CONFIGS['OLMo-7B']['model_name_mapping']
ROW_ORDER = MODEL_CONFIGS['OLMo-7B']['row_order']
TRAIN_LOSS_FILE_MAPPING = MODEL_CONFIGS['OLMo-7B']['train_loss_file_mapping']
TRAIN_LOSS_MODEL_TO_TABLE = MODEL_CONFIGS['OLMo-7B']['train_loss_model_to_table']

TARGET_STEP = 3000

def detect_model_from_path(eval_folder):
    """Detect model type from eval folder path."""
    if 'OLMo-1b' in eval_folder or 'OLMo-1B' in eval_folder:
        return 'OLMo-1B'
    elif 'OLMo-7b' in eval_folder or 'OLMo-7B' in eval_folder:
        return 'OLMo-7B'
    else:
        # Default to OLMo-7B for backward compatibility
        return 'OLMo-7B'

def read_csv_file(file_path):
    """Read CSV file and return as list of dictionaries."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Try to detect if file has quotes
            sample = f.read(1024)
            f.seek(0)
            if sample.startswith('"'):
                reader = csv.DictReader(f, quoting=csv.QUOTE_ALL)
            else:
                reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
    return data

def normalize_column_name(col_name):
    """Remove quotes from column name for comparison."""
    return col_name.strip('"')

def find_value_at_step(data, step_col, target_step, model_col_prefix):
    """Find value at target step for a specific model."""
    # First, find the column name that matches the model prefix
    matching_col = None
    if data:
        for col_name in data[0].keys():
            col_clean = normalize_column_name(col_name)
            if col_clean.startswith(model_col_prefix) and not col_clean.endswith('__MIN') and not col_clean.endswith('__MAX'):
                matching_col = col_name
                break
    
    if not matching_col:
        return None
    
    # Now find the value at target step
    for row in data:
        step = normalize_column_name(row.get(step_col, ''))
        try:
            if int(step) == target_step:
                value = row.get(matching_col, '')
                if value and value.strip():
                    try:
                        value_clean = normalize_column_name(str(value))
                        return float(value_clean)
                    except (ValueError, TypeError):
                        pass
        except (ValueError, TypeError):
            continue
    return None

def identify_metric_from_file(file_path):
    """Identify which metric a CSV file contains by examining column names."""
    data = read_csv_file(file_path)
    if not data:
        return None
    
    # Check first row's column names
    for col_name in data[0].keys():
        col_clean = normalize_column_name(col_name).lower()
        if 'arc_easy' in col_clean:
            return 'arc_easy_acc'
        elif 'copa' in col_clean:
            return 'copa_acc'
        elif 'sciq' in col_clean:
            return 'sciq_acc'
        elif 'hellaswag' in col_clean:
            return 'hellaswag_len_norm'
        elif 'wikitext' in col_clean:
            return 'wikitext_ppl'
        elif 'pile' in col_clean and 'perplexity' in col_clean:
            return 'pile_ppl'
        elif 'c4' in col_clean and 'perplexity' in col_clean:
            return 'c4_ppl'
    return None

def extract_eval_metrics(eval_folder, target_step, model_config):
    """Extract evaluation metrics from eval folder CSV files."""
    metrics = defaultdict(lambda: defaultdict(dict))
    model_name_mapping = model_config['model_name_mapping']
    
    # Get all CSV files in eval folder
    csv_files = glob.glob(os.path.join(eval_folder, '*.csv'))
    
    # Map each file to its metric type
    file_metric_map = {}
    for file_path in csv_files:
        metric_name = identify_metric_from_file(file_path)
        if metric_name:
            file_metric_map[metric_name] = file_path
    
    # Extract metrics from each file
    for metric_name, file_path in file_metric_map.items():
        data = read_csv_file(file_path)
        if not data:
            continue
        
        step_col = 'Step'
        # Extract metrics for each model
        for model_prefix, table_name in model_name_mapping.items():
            value = find_value_at_step(data, step_col, target_step, model_prefix)
            if value is not None:
                metrics[table_name][metric_name] = value
    
    return metrics

def extract_train_losses(logs_dir, eval_folder, target_step, model_config):
    """Extract training losses from CSV files in logs directory (excluding eval folder)."""
    train_losses = {}
    train_loss_file_mapping = model_config['train_loss_file_mapping']
    train_loss_model_to_table = model_config['train_loss_model_to_table']
    csv_subfolder = model_config.get('csv_subfolder', '')
    
    # Get all CSV files in logs directory (parent of eval folder)
    if eval_folder.endswith('eval') or eval_folder.endswith('eval/'):
        logs_dir = os.path.dirname(eval_folder)
    else:
        logs_dir = os.path.dirname(os.path.dirname(eval_folder))
    
    # Determine CSV directory based on model config
    if csv_subfolder:
        csv_dir = os.path.join(logs_dir, csv_subfolder)
    else:
        # For OLMo-7B, CSV files are in logs/ directory (one level up)
        parent_logs_dir = os.path.dirname(logs_dir) if os.path.basename(logs_dir) == 'OLMo-7b' else logs_dir
        csv_dir = parent_logs_dir
    
    if not os.path.exists(csv_dir):
        print(f"Warning: CSV directory not found: {csv_dir}")
        return train_losses
    
    csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))
    
    # Filter out eval folder
    eval_folder_abs = os.path.abspath(eval_folder)
    csv_files = [f for f in csv_files if 'eval' not in os.path.abspath(f)]
    
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        # Find matching model prefix based on filename
        # Sort by length (longest first) to match more specific prefixes first
        sorted_mappings = sorted(train_loss_file_mapping.items(), key=lambda x: len(x[0]), reverse=True)
        for file_prefix, model_prefix in sorted_mappings:
            if filename.startswith(file_prefix):
                # Get the table name for this model prefix
                table_name = train_loss_model_to_table.get(model_prefix)
                if not table_name:
                    continue
                    
                data = read_csv_file(csv_file)
                if not data:
                    continue
                
                step_col = 'Step'
                # Look for train/CrossEntropyLoss column for this model
                train_col_prefix = f"{model_prefix} - train/CrossEntropyLoss"
                value = find_value_at_step(data, step_col, target_step, train_col_prefix)
                if value is not None:
                    train_losses[table_name] = value
                break
    
    return train_losses

def print_table(metrics, train_losses, row_order, display_name_mapping=None, target_step=3000):
    """Print the evaluation table in the requested format."""
    # Prepare data for each row
    rows_data = {}
    
    for row_name in row_order:
        rows_data[row_name] = {
            'train_loss': train_losses.get(row_name, None),
            'wikitext_ppl': metrics[row_name].get('wikitext_ppl', None),
            'c4_ppl': metrics[row_name].get('c4_ppl', None),
            'pile_ppl': metrics[row_name].get('pile_ppl', None),
            'copa_acc': metrics[row_name].get('copa_acc', None),
            'arc_easy_acc': metrics[row_name].get('arc_easy_acc', None),
            'sciq_acc': metrics[row_name].get('sciq_acc', None),
            'hellaswag_len_norm': metrics[row_name].get('hellaswag_len_norm', None),
        }
    
    # Note: mxfp4 data is not provided yet, so it will remain empty (all None values)
    # Four Over Six data comes from auto-reverse model (already mapped above)
    
    # Calculate averages
    for row_name in row_order:
        data = rows_data[row_name]
        # Avg PPL = average of wikitext, c4, pile
        ppls = [data['wikitext_ppl'], data['c4_ppl'], data['pile_ppl']]
        ppls = [p for p in ppls if p is not None]
        data['avg_ppl'] = sum(ppls) / len(ppls) if ppls else None
        
        # Avg Acc = average of copa, arc_easy, sciq, hellaswag
        accs = [data['copa_acc'], data['arc_easy_acc'], data['sciq_acc'], data['hellaswag_len_norm']]
        accs = [a for a in accs if a is not None]
        data['avg_acc'] = sum(accs) / len(accs) if accs else None
    
    # Print table - Part 1: Perplexity metrics
    print("=" * 80)
    print(f"Evaluation Results at Step {target_step}")
    print("=" * 80)
    print()
    print(f"{'Model':<25} {'Train Loss':<12} {'WikiText':<12} {'C4':<12} {'Pile':<12} {'Avg PPL':<12}")
    print("-" * 80)
    
    for row_name in row_order:
        data = rows_data[row_name]
        
        # Format row name with special formatting
        if display_name_mapping and row_name in display_name_mapping:
            display_name = display_name_mapping[row_name]
        elif row_name == 'mxfp4':
            display_name = 'mxfp4 ($S_{\\text{max}}$)'
        elif row_name == 'Four Over Six':
            display_name = 'Four Over Six'
        elif row_name == 'Fixed Exp-1':
            display_name = 'Fixed Exp-1 ($S/2$)'
        elif row_name == 'Fixed Exp-2':
            display_name = 'Fixed Exp-2 ($S/4$)'
        elif row_name == 'Mixed':
            display_name = 'Mixed (S + S/4)'
        elif row_name == 'TRACE':
            display_name = 'TRACE (Ours)'
        else:
            display_name = row_name
        
        # Format values
        train_loss_str = f"{data['train_loss']:.4f}" if data['train_loss'] is not None else "-"
        wikitext_str = f"{data['wikitext_ppl']:.2f}" if data['wikitext_ppl'] is not None else "-"
        c4_str = f"{data['c4_ppl']:.2f}" if data['c4_ppl'] is not None else "-"
        pile_str = f"{data['pile_ppl']:.2f}" if data['pile_ppl'] is not None else "-"
        avg_ppl_str = f"{data['avg_ppl']:.2f}" if data['avg_ppl'] is not None else "-"
        
        print(f"{display_name:<25} {train_loss_str:<12} {wikitext_str:<12} {c4_str:<12} {pile_str:<12} {avg_ppl_str:<12}")
    
    print()
    print("=" * 80)
    print(f"{'Model':<25} {'COPA':<12} {'ARC(E)':<12} {'SciQ':<12} {'HellaSwag':<12} {'Avg Acc':<12}")
    print("-" * 80)
    
    for row_name in row_order:
        data = rows_data[row_name]
        
        # Format row name
        if display_name_mapping and row_name in display_name_mapping:
            display_name = display_name_mapping[row_name]
        elif row_name == 'mxfp4':
            display_name = 'mxfp4 ($S_{\\text{max}}$)'
        elif row_name == 'Four Over Six':
            display_name = 'Four Over Six'
        elif row_name == 'Fixed Exp-1':
            display_name = 'Fixed Exp-1 ($S/2$)'
        elif row_name == 'Fixed Exp-2':
            display_name = 'Fixed Exp-2 ($S/4$)'
        elif row_name == 'Mixed':
            display_name = 'Mixed (S + S/4)'
        elif row_name == 'TRACE':
            display_name = 'TRACE (Ours)'
        else:
            display_name = row_name
        
        # Format values (convert accuracy to percentage if needed, or keep as is)
        copa_str = f"{data['copa_acc']*100:.2f}" if data['copa_acc'] is not None else "-"
        arc_str = f"{data['arc_easy_acc']*100:.2f}" if data['arc_easy_acc'] is not None else "-"
        sciq_str = f"{data['sciq_acc']*100:.2f}" if data['sciq_acc'] is not None else "-"
        hellaswag_str = f"{data['hellaswag_len_norm']*100:.2f}" if data['hellaswag_len_norm'] is not None else "-"
        avg_acc_str = f"{data['avg_acc']*100:.2f}" if data['avg_acc'] is not None else "-"
        
        print(f"{display_name:<25} {copa_str:<12} {arc_str:<12} {sciq_str:<12} {hellaswag_str:<12} {avg_acc_str:<12}")
    
    print("=" * 80)

def main():
    if len(sys.argv) > 1:
        eval_folder = sys.argv[1]
    else:
        # Default to logs/OLMo-7b/eval
        eval_folder = 'logs/OLMo-7b/eval'
    
    if not os.path.exists(eval_folder):
        print(f"Error: Eval folder not found: {eval_folder}")
        sys.exit(1)
    
    # Detect model type from path
    model_type = detect_model_from_path(eval_folder)
    model_config = MODEL_CONFIGS[model_type]
    
    logs_dir = os.path.dirname(eval_folder) if eval_folder.endswith('eval') or eval_folder.endswith('eval/') else os.path.dirname(os.path.dirname(eval_folder))
    
    # Determine CSV directory for logging
    csv_subfolder = model_config.get('csv_subfolder', '')
    if csv_subfolder:
        csv_dir = os.path.join(logs_dir, csv_subfolder)
    else:
        csv_dir = os.path.dirname(logs_dir) if os.path.basename(logs_dir) == 'OLMo-7b' else logs_dir
    
    print(f"Detected model type: {model_type}")
    print(f"Extracting metrics from: {eval_folder}")
    print(f"Looking for training losses in: {csv_dir}")
    
    # Get target step from model config
    target_step = model_config.get('target_step', TARGET_STEP)
    print(f"Target step: {target_step}")
    print()
    
    # Extract evaluation metrics
    metrics = extract_eval_metrics(eval_folder, target_step, model_config)
    
    # Extract training losses
    train_losses = extract_train_losses(logs_dir, eval_folder, target_step, model_config)
    
    # Set empty values for rows that don't have data (TRACE models for OLMo-1B)
    row_order = model_config['row_order']
    for row_name in row_order:
        if row_name not in metrics:
            metrics[row_name] = {}
        if row_name not in train_losses:
            train_losses[row_name] = None
    
    # Create display name mapping for OLMo-1B
    display_name_mapping = None
    if model_type == 'OLMo-1B':
        display_name_mapping = {
            'BF16': 'BF16',
            'mxfp4 (Linear)': 'mxfp4 (Linear)',
            'mxfp4 (A+L)': 'mxfp4 (A+L)',
            'TRACE (Linear)': 'TRACE (Linear)',
            'TRACE (A+L)': 'TRACE (A+L)',
        }
    
    # Print table
    print_table(metrics, train_losses, row_order, display_name_mapping, target_step)

if __name__ == "__main__":
    main()

