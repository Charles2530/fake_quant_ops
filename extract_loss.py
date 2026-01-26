#!/usr/bin/env python3
"""
Extract step and CrossEntropyLoss from CSV log files.
When a step has multiple CrossEntropyLoss values, select the one from the run
that restarts from the nearest hundred (e.g., for step 100-199, use the run starting from step 100).
"""

import csv
import os
import sys


def find_crossentropy_columns(header):
    """Find all CrossEntropyLoss columns (excluding MIN and MAX)."""
    crossentropy_cols = []
    for i, col in enumerate(header):
        if 'CrossEntropyLoss' in col and '__MIN' not in col and '__MAX' not in col:
            crossentropy_cols.append(i)
    return crossentropy_cols


def find_start_step_for_column(csv_file, col_idx):
    """Find the first step where this column has a non-empty value."""
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) > col_idx and row[col_idx].strip():
                try:
                    step = int(row[0].strip('"'))
                    return step
                except (ValueError, IndexError):
                    continue
    return None


def extract_loss_from_csv(csv_file, output_file):
    """Extract step and CrossEntropyLoss from CSV file."""
    # Read the CSV file
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        # Find all CrossEntropyLoss columns
        crossentropy_cols = find_crossentropy_columns(header)
        
        if not crossentropy_cols:
            print(f"Warning: No CrossEntropyLoss columns found in {csv_file}")
            return
        
        # Find the start step for each column
        col_start_steps = {}
        for col_idx in crossentropy_cols:
            start_step = find_start_step_for_column(csv_file, col_idx)
            if start_step is not None:
                col_start_steps[col_idx] = start_step
        
        # Read all data
        data = []
        for row in reader:
            if not row or not row[0].strip():
                continue
            try:
                step = int(row[0].strip('"'))
                data.append((step, row))
            except (ValueError, IndexError):
                continue
        
        # Extract loss for each step
        results = []
        for step, row in data:
            # Find the appropriate column for this step
            # Select the column that starts from the nearest hundred (e.g., for step 100-199, use run starting from step 100)
            step_hundred = (step // 100) * 100
            
            selected_col = None
            selected_value = None
            
            # Collect all available values for this step
            available_values = []
            for col_idx in crossentropy_cols:
                if len(row) > col_idx and row[col_idx].strip():
                    value = row[col_idx].strip('"')
                    if col_idx in col_start_steps:
                        start_step = col_start_steps[col_idx]
                        available_values.append((col_idx, value, start_step))
                    else:
                        available_values.append((col_idx, value, None))
            
            if not available_values:
                continue
            
            # If only one value, use it
            if len(available_values) == 1:
                selected_value = available_values[0][1]
            else:
                # Multiple values: prefer the one from the run that starts at the step's hundred
                # For step 100-199, prefer run starting at 100
                # For step 200-299, prefer run starting at 200, etc.
                best_col = None
                best_value = None
                best_start = None
                
                # First priority: exact match with step_hundred
                for col_idx, value, start_step in available_values:
                    if start_step == step_hundred:
                        best_col = col_idx
                        best_value = value
                        best_start = start_step
                        break
                
                # Second priority: closest start step that is <= step_hundred
                if best_col is None:
                    min_diff = float('inf')
                    for col_idx, value, start_step in available_values:
                        if start_step is not None and start_step <= step_hundred:
                            diff = step_hundred - start_step
                            if diff < min_diff:
                                min_diff = diff
                                best_col = col_idx
                                best_value = value
                                best_start = start_step
                
                # Third priority: any available value (fallback)
                if best_col is None:
                    best_value = available_values[0][1]
                
                selected_value = best_value
            
            if selected_value:
                results.append((step, selected_value))
        
        # Write to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Step\tCrossEntropyLoss\n")
            for step, loss in results:
                f.write(f"{step}\t{loss}\n")
        
        print(f"Extracted {len(results)} steps from {csv_file}")
        print(f"Output written to {output_file}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_loss.py <csv_file> [output_file]")
        print("  csv_file: Path to the CSV log file")
        print("  output_file: Optional output file path (default: <csv_file>.txt)")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    if not os.path.exists(csv_file):
        print(f"Error: File not found: {csv_file}")
        sys.exit(1)
    
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]
    else:
        # Default output file: same name as input but with .txt extension
        output_file = os.path.splitext(csv_file)[0] + '.txt'
    
    extract_loss_from_csv(csv_file, output_file)


if __name__ == '__main__':
    main()

