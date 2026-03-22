#!/bin/bash
set -e

# Download model and soundfont if not present
echo "Downloading required files..."
python -c "
from pathlib import Path
from huggingface_hub import hf_hub_download

# Download model
model_path = Path('deploy/model.pth')
if not model_path.exists():
    print('Downloading model...')
    hf_hub_download(repo_id='NicolaDiSalvatore/BachGen1.0', filename='model.pth', local_dir='deploy')

# Download soundfont
sf_path = Path('resources/052_Florestan_Ahh_Choir.sf2')
if not sf_path.exists():
    print('Downloading soundfont...')
    hf_hub_download(repo_id='NicolaDiSalvatore/florestan-ahh-choir-soundfont', filename='052_Florestan_Ahh_Choir.sf2', repo_type='dataset', local_dir='resources')
print('Download complete!')
"

# Start the application
exec uvicorn api.main:app --host 0.0.0.0 --port "$PORT"