import sys
from pathlib import Path

# Add root directory to python path so it can import backend_app and 'backend'
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

# Import Flask app instance
from backend_app import app
