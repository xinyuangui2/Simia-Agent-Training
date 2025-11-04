#!/bin/bash

RAGEN_VERSION=b6146567050d521f8d32b661b15c975a4872cccc

# Exit on error
set -e

# Function to check if CUDA is available
check_cuda() {
    if command -v nvcc --version &> /dev/null; then
        echo "CUDA GPU detected"
        return 0
    else
        echo "No CUDA GPU detected"
        return 1
    fi
}

# Function to check if conda is available
check_conda() {
    if command -v conda &> /dev/null; then
        echo "Conda is available"
        return 0
    else
        echo "Conda is not installed. Please install Conda first."
        return 1
    fi
}

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print step with color
print_step() {
    echo -e "${BLUE}[Step] ${1}${NC}"
}

# Main installation process
main() {
    # Check prerequisites
    check_conda || exit 1
    
    # Create and activate conda environment
    # if not exists, create it
    print_step "Creating conda environment 'simia-rl' with Python 3.12..."
    conda create python=3.12 -n simia-rl -y

    # Need to source conda for script environment
    eval "$(conda shell.bash hook)"
    conda activate simia-rl
    
    # Clone repository
    print_step "Cloning ragen repository..."
    git clone https://github.com/RAGEN-AI/RAGEN.git ragen
    cd ragen
    git checkout $RAGEN_VERSION

    # Install package in editable mode
    print_step "setting up verl..."
    git submodule init
    git submodule update
    cd verl
    pip install -e . --no-dependencies # we put dependencies in RAGEN/requirements.txt
    cd ..

    pip install azureml-mlflow mlflow

    # Install vLLM
    print_step "Installing vLLM..."
    pip install vllm==0.8.5
    
    # Install package in editable mode
    print_step "Installing ragen package..."
    pip install -e .

    
    # Install PyTorch with CUDA if available
    print_step "Installing PyTorch with CUDA support..."
    pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
        
    print_step "Installing flash-attention..."
    pip cache purge
    pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
    
    # Install remaining requirements
    print_step "Installing additional requirements..."
    pip install -r requirements.txt

    cd ..
    pip install -r remaining_requirements.txt

    # Remove ragen
    print_step "Removing ragen repository..."
    rm -rf ragen

    # Reinstall local ragen and verl in editable mode
    print_step "Reinstalling ragen and verl in editable mode..."
    pip install -e subtrees/verl -e subtrees/ragen

    echo -e "${GREEN}Installation completed successfully!${NC}"


    # Install specific versions of vLLM
    print_step "Installing vLLM..."
    pip install vllm==0.8.5

    # Install specific versions of ray and opentelemetry
    print_step "Installing ray 2.49.1..."
    pip install ray==2.49.1

    print_step "Installing opentelemetry 1.26.0..."
    pip install opentelemetry-api==1.26.0 opentelemetry-sdk==1.26.0
}

# Run main installation
main