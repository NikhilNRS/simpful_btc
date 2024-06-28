#!/bin/bash

# Create a directory to store the JSON files if it doesn't already exist
output_dir="json_notebooks"
mkdir -p "$output_dir"

# List of Jupyter notebook files to be converted
notebooks=(
    "01_data_integration.ipynb"
    "02_data_inspection.ipynb"
    "03_exploratory_data_analysis.ipynb"
    "04_PCA_&_Transform_To_Modelling_Format_Daily.ipynb"
    "04_PCA_&_Transform_To_Modelling_Format_Hourly.ipynb"
    "04_PCA_&_Transform_To_Modelling_Format_Hourly_Daily.ipynb"
    "05_Feature_analysis_Daily.ipynb"
    "05_Feature_analysis_Hourly.ipynb"
    "06_RNN_model_BTC_Daily.ipynb"
    "06_RNN_model_BTC_Hourly.ipynb"
    "07_HyperParameterTuning_LSTM_RNN_Daily.ipynb"
    "07_HyperParameterTuning_LSTM_RNN_Hourly.ipynb"
    "08_Fuzzy_Basic_Daily.ipynb"
    "08_Fuzzy_Basic_Hourly.ipynb"
    "09_Fuzzy_Hypertuned_Daily.ipynb"
    "09_Fuzzy_Hypertuned_Hourly.ipynb"
    "10_Fuzzy_GP_Daily.ipynb"
    "10_Fuzzy_GP_Daily_Debug.ipynb"
    "11_Model_Inspection.ipynb"
    "Appendix_Simple_Example.ipynb"
    "Example_Of_Simpful_Functionality.ipynb"
)

# Convert each notebook to raw JSON and save in the output directory
for notebook in "${notebooks[@]}"; do
    # Ensure the notebook file exists
    if [ -f "$notebook" ]; then
        jq . "$notebook" > "$output_dir/$(basename "$notebook" .ipynb).json"
        echo "Converted $notebook to raw JSON format and saved to $output_dir/$(basename "$notebook" .ipynb).json"
    else
        echo "Notebook $notebook not found!"
    fi
done

echo "Conversion to raw JSON completed."
