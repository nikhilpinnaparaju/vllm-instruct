# Repository README

This repository contains code for processing JSON data using a language model and distributing the processing across multiple parts. Below is an overview of the repository structure, the Python code for data processing, and bash scripts for distributing the workload.

## Repository Structure

- `README.md`: This file, providing an overview of the repository.
- `data_processing.py`: Python script for processing JSON data using a language model.
- `distributor.sh`: Bash script for distributing the data processing workload across multiple parts.
- `vllm_processor.sh`: Bash script to process a single part of the data using a language model.
- `requirements.txt`: List of Python dependencies required to run the code.

## Data Processing

The `data_processing.py` script utilizes a language model to generate responses for user questions. It reads input data from a JSON file and produces processed output in various formats.

### Usage:

```bash
python data_processing.py input_json_file
```
- input_json_file: Path to the input JSON file containing user questions and inputs.

## Distribution Script
The `distributor.sh` script distributes the data processing workload across multiple parts, allowing parallel processing for improved efficiency. It splits the input JSON file into several parts and submits individual processing jobs for each part.
```bash
./distributor.sh input_json_file output_directory num_parts
```
- `input_json_file`: Path to the input JSON file containing user questions and inputs.
- `output_directory`: Directory to store the processed output.
- `num_parts`: Number of parts to split the input data into for parallel processing.

## Example Usage

```bash
./distributor.sh input.json processed_data 4
```
This command splits the input.json file into 4 parts and processes each part in parallel, storing the processed output in the processed_data directory.

For more detailed information on each script and its usage, refer to the comments within the script files.