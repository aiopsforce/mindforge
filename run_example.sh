#!/bin/bash

# --- Configuration ---
PYTHON_VERSION_MAJOR=3
PYTHON_VERSION_MINOR=12
VENV_DIR=".venv"
EXAMPLE_SCRIPT="examples/full_example.py"

# --- Helper Functions ---
check_python_version() {
    echo "Checking Python version..."
    if command -v python3 &>/dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &>/dev/null; then
        PYTHON_CMD="python"
    else
        echo "ERROR: Python not found."
        exit 1
    fi

    # Get actual version
    ACTUAL_MAJOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.major)')
    ACTUAL_MINOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.minor)')

    if [ "$ACTUAL_MAJOR" -lt "$PYTHON_VERSION_MAJOR" ] || ([ "$ACTUAL_MAJOR" -eq "$PYTHON_VERSION_MAJOR" ] && [ "$ACTUAL_MINOR" -lt "$PYTHON_VERSION_MINOR" ]); then
        echo "ERROR: Python version $PYTHON_VERSION_MAJOR.$PYTHON_VERSION_MINOR or higher is required. Found $ACTUAL_MAJOR.$ACTUAL_MINOR."
        echo "Please install or switch to a compatible Python version."
        exit 1
    else
        echo "Python version $ACTUAL_MAJOR.$ACTUAL_MINOR found. ($PYTHON_CMD)"
    fi
}

create_virtual_env() {
    if [ ! -d "$VENV_DIR" ]; then
        echo "Creating virtual environment in $VENV_DIR..."
        $PYTHON_CMD -m venv "$VENV_DIR"
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to create virtual environment."
            exit 1
        fi
        echo "Virtual environment created."
    else
        echo "Virtual environment already exists in $VENV_DIR."
    fi
}

install_dependencies() {
    echo "Activating virtual environment..."
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to activate virtual environment."
        exit 1
    fi

    echo "Installing dependencies using 'pip install -e .'..."
    pip install -e .
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to install dependencies."
        echo "Please check for any errors above."
        exit 1
    fi
    echo "Dependencies installed successfully."
}

run_example_script() {
    echo ""
    echo "---------------------------------------------------------------------"
    echo "IMPORTANT: API Key Configuration"
    echo "---------------------------------------------------------------------"
    echo "The MindForge example script requires API keys for the AI models you"
    echo "intend to use (OpenAI, Azure OpenAI, or Ollama settings)."
    echo ""
    echo "Please ensure the following environment variables are set if you plan"
    echo "to use the respective services:"
    echo ""
    echo "For OpenAI:"
    echo "  export OPENAI_API_KEY="your_openai_api_key""
    echo ""
    echo "For Azure OpenAI:"
    echo "  export AZURE_OPENAI_KEY="your_azure_api_key""
    echo "  export AZURE_OPENAI_ENDPOINT="your_azure_endpoint""
    echo "  (Optionally, configure Azure settings in mindforge/config.py or full_example.py)"
    echo ""
    echo "For Ollama (if not using the default http://localhost:11434):"
    echo "  The example script can be configured to use a specific Ollama URL."
    echo "  You might need to modify 'config.model.ollama_base_url' in the example"
    echo "  or AppConfig if your Ollama instance is not at the default."
    echo ""
    echo "You can set these variables in your current shell session or add them"
    echo "to your shell's profile (e.g., .bashrc, .zshrc)."
    echo "---------------------------------------------------------------------"
    echo ""
    read -r -p "Have you configured the necessary API keys/environment variables? (yes/no): " CONFIRMED

    if [[ "$CONFIRMED" != "yes" ]]; then
        echo "Please set the required environment variables and re-run the script."
        exit 0
    fi

    echo "Running the example script: $EXAMPLE_SCRIPT..."
    echo "Ensure the virtual environment ($VENV_DIR) is active."
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    $PYTHON_CMD "$EXAMPLE_SCRIPT"
}

# --- Main Script ---
echo "--- MindForge Example Setup & Run Script ---"

check_python_version
create_virtual_env
install_dependencies
run_example_script

echo ""
echo "--- Script Finished ---"
