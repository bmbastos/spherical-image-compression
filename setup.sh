# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python not found. Please install Python and try again."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check for pip updates
if pip list --outdated | grep -q "^pip "; then
    echo "Updating pip..."
    python3 -m pip install --upgrade pip
else
    echo "Pip is up to date."
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Done
echo
echo "Environment successfully set up!"
exit 0