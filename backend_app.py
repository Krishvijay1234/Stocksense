from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

BACKEND_PATH = Path(__file__).resolve().parent / 'New folder' / 'flask_backend.py'
SPEC = spec_from_file_location('stocksense_backend', BACKEND_PATH)
BACKEND_MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(BACKEND_MODULE)
app = BACKEND_MODULE.app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
