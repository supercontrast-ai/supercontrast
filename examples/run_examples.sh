#!/bin/bash

# Azure 
export AZURE_TEXT_ANALYTICS_ENDPOINT="https://supercontrast-ai-services-instance.cognitiveservices.azure.com/"
export AZURE_TEXT_ANALYTICS_KEY="365bcd442f9e4d4b95577a320e833726"
export AZURE_TRANSLATOR_REGION="eastus"
export AZURE_VISION_ENDPOINT="https://supercontrast-ai-services-instance.cognitiveservices.azure.com/"
export AZURE_VISION_KEY="365bcd442f9e4d4b95577a320e833726"

# GCP
export GCP_API_KEY="AIzaSyC97eE6fM8GWecwxazrTp02ZdedybIhTkY"

# AWS

echo "Running examples.py"
python examples.py
echo "Done"