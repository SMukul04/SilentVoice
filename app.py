import sys
import os

# Add project root to python path to resolve backend package imports
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from backend.app import app

if __name__ == '__main__':
    print("Launching SilentVoice Server from Root Directory...")
    app.run(debug=True, host='0.0.0.0', port=5000)
