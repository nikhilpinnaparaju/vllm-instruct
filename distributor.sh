#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <input_json_file> <output_directory> <num_parts>"
    exit 1
fi

# Assign input parameters to variables
input_file=$1
output_directory=$2
num_parts=$3

# Create output directory if it doesn't exist
cd /weka/home-nikhilp/instruct-datafiltering
mkdir -p "$output_directory"

# Calculate the number of datapoints per subfile
split -n l/$num_parts -d $input_file "$output_directory/part"
echo $PWD
# Loop through each part and submit an sbatch job
for ((i=0; i<num_parts; i++)); do
    formatted_i=$(printf "%02d" $i)
    sbatch /weka/home-nikhilp/instruct-datafiltering/vllm_processor.sh "$output_directory/part$formatted_i"
    echo "$output_directory/part$formatted_i"
done

echo "Jobs submitted successfully."
