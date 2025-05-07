#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# Check available space on D: drive (Windows specific)
free_space=$(df -h D: | awk 'NR==2 {print $4}' | sed 's/G//')
if (( $(echo "$free_space < 100" | bc -l) )); then
    echo -e "${RED}Warning: Less than 100GB free space on D: drive${NC}"
    echo -e "${YELLOW}Available space: ${free_space}GB${NC}"
    echo -e "${YELLOW}Recommended space: 100GB${NC}"
    echo -e "${YELLOW}Do you want to continue? (y/n)${NC}"
    read -r continue_install
    if [ "$continue_install" != "y" ]; then
        exit 1
    fi
fi

# Create models directory
if [ ! -d "D:\Models" ]; then
    echo -e "${YELLOW}Creating D:\Models directory...${NC}"
    mkdir -p "D:\Models" || {
        echo -e "${RED}Error: Failed to create D:\Models directory. Check permissions.${NC}"
        exit 1
    }
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

# Function to download models by category
download_category() {
    local category=$1
    echo -e "${BLUE}Downloading ${category} models...${NC}"
    python download_specialized_models.py --category "$category"
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Successfully downloaded ${category} models${NC}"
        return 0
    else
        echo -e "${RED}Failed to download some ${category} models${NC}"
        return 1
    fi
}

# Display available categories
echo -e "\n${YELLOW}Available model categories:${NC}"
python -c "from download_specialized_models import MODELS; print('\n'.join(MODELS.keys()))"

# Ask user which categories to download
echo -e "\n${YELLOW}Download options:${NC}"
echo "1) All models (requires ~90GB)"
echo "2) Select specific categories"
echo "3) Essential models only (vision, text, audio)"
read -p "Choose an option (1-3): " download_option

case $download_option in
    1)
        echo -e "${YELLOW}Downloading all models...${NC}"
        python download_specialized_models.py --category all
        ;;
    2)
        echo -e "${YELLOW}Enter categories to download (space-separated):${NC}"
        read -r categories
        for category in $categories; do
            download_category "$category"
        done
        ;;
    3)
        echo -e "${YELLOW}Downloading essential models...${NC}"
        for category in vision text audio; do
            download_category "$category"
        done
        ;;
    *)
        echo -e "${RED}Invalid option${NC}"
        exit 1
        ;;
esac

# Verify installations
echo -e "\n${YELLOW}Verifying installations...${NC}"
for category in $(python -c "from download_specialized_models import MODELS; print(' '.join(MODELS.keys()))"); do
    if [ -d "D:\Models\\$category" ]; then
        model_count=$(ls -1 "D:\Models\\$category" | wc -l)
        echo -e "${GREEN}${category}: ${model_count} models installed${NC}"
    else
        echo -e "${RED}${category}: No models found${NC}"
    fi
done

# Test loading some models
echo -e "\n${YELLOW}Testing model loading...${NC}"
echo -e "${BLUE}Testing vision model...${NC}"
python download_specialized_models.py --load --model-name YOLO

echo -e "${BLUE}Testing text model...${NC}"
python download_specialized_models.py --load --model-name qna

echo -e "${BLUE}Testing audio model...${NC}"
python download_specialized_models.py --load --model-name whisper

if [ "$use_venv" = "y" ]; then
    echo -e "\n${YELLOW}Virtual environment information:${NC}"
    echo -e "${GREEN}To activate: source .venv/Scripts/activate${NC}"
    echo -e "${GREEN}To deactivate: deactivate${NC}"
fi

echo -e "\n${GREEN}Setup complete!${NC}"
echo -e "${YELLOW}Models are stored in D:\Models${NC}"
echo -e "${YELLOW}Check README_MODELS.md for usage instructions${NC}"