#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting specialized models setup...${NC}"

# Check if running on Windows
if [[ "$OSTYPE" != "msys" && "$OSTYPE" != "cygwin" && "$OSTYPE" != "win32" ]]; then
    echo -e "${RED}Error: This script is designed for Windows systems with D: drive${NC}"
    exit 1
fi

# Check if D: drive exists
if [ ! -d "D:" ]; then
    echo -e "${RED}Error: D: drive not found${NC}"
    exit 1
fi

# Check if we can write to D:\Models
if [ ! -d "D:\Models" ]; then
    echo -e "${YELLOW}Creating D:\Models directory...${NC}"
    mkdir -p "D:\Models" || {
        echo -e "${RED}Error: Failed to create D:\Models directory. Check permissions.${NC}"
        exit 1
    }
fi

# Check available space on D: drive (Windows specific)
free_space=$(df -h D: | awk 'NR==2 {print $4}' | sed 's/G//')
if (( $(echo "$free_space < 20" | bc -l) )); then
    echo -e "${RED}Warning: Less than 20GB free space on D: drive${NC}"
    echo -e "${YELLOW}Available space: ${free_space}GB${NC}"
    echo -e "${YELLOW}Do you want to continue? (y/n)${NC}"
    read -r continue_install
    if [ "$continue_install" != "y" ]; then
        exit 1
    fi
fi

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo -e "${RED}Error: pip is not installed. Please install pip first.${NC}"
    exit 1
fi

# Create and activate virtual environment (optional)
echo -e "${YELLOW}Would you like to create a virtual environment? (y/n)${NC}"
read -r use_venv

if [ "$use_venv" = "y" ]; then
    echo -e "${GREEN}Creating virtual environment...${NC}"
    python -m venv .venv
    
    # Activate virtual environment
    source .venv/Scripts/activate
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to create/activate virtual environment${NC}"
        exit 1
    fi
fi

# Install dependencies
echo -e "${GREEN}Installing requirements...${NC}"
pip install -r requirements_specialized.txt

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to install requirements${NC}"
    exit 1
fi

# Download models
echo -e "${GREEN}Downloading specialized models to D:\Models...${NC}"

# Audio models
echo -e "${YELLOW}Downloading Audio models...${NC}"
python download_specialized_models.py --category audio

# Vision models
echo -e "${YELLOW}Downloading Vision models...${NC}"
python download_specialized_models.py --category vision

# Multimodal models
echo -e "${YELLOW}Downloading Multimodal models...${NC}"
python download_specialized_models.py --category multimodal

# Check if models directory exists and has content
if [ -d "D:\Models" ] && [ "$(ls -A 'D:\Models')" ]; then
    echo -e "${GREEN}Setup completed successfully!${NC}"
    echo -e "${GREEN}Models are stored in D:\Models${NC}"
else
    echo -e "${RED}Setup failed: Models directory is empty or doesn't exist${NC}"
    exit 1
fi

if [ "$use_venv" = "y" ]; then
    echo -e "${YELLOW}To activate the virtual environment in the future, use:${NC}"
    echo -e "${GREEN}source .venv/Scripts/activate${NC}"
fi

echo -e "${YELLOW}Checking models...${NC}"
# Test loading each category
python download_specialized_models.py --load --model-name wav2vec2
python download_specialized_models.py --load --model-name vit
python download_specialized_models.py --load --model-name clip

echo -e "${GREEN}Installation complete!${NC}"
echo -e "${YELLOW}Models are stored in D:\Models${NC}"
echo -e "${YELLOW}You can now use these models in your applications${NC}"