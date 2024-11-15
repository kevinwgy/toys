#!/bin/bash

# Define input and output files
input_file="results_clean.txt"
training_file="training_data.txt"
validation_file="validation_data.txt"

# Count the number of rows in the input file
total_rows=$(wc -l < "$input_file")

# Calculate the number of rows for training and validation
training_rows=$((total_rows * 4 / 5))
validation_rows=$((total_rows - training_rows))

# Split the file into training and validation datasets
head -n "$training_rows" "$input_file" > "$training_file"
tail -n "$validation_rows" "$input_file" > "$validation_file"

echo "Training data saved to $training_file"
echo "Validation data saved to $validation_file"
