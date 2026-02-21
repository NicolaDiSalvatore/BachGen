"""
This script is made to convert each chorale sequnence of dim (batch_size, sequence_length, 4) into a MIDI file.
"""

import json
import os
from pathlib import Path

from rendering.midi import sequences_to_midi

project_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
json_file = project_dir / "data" / "raw" / "Jsb16thSeparated.json"

with open(json_file, 'r') as f:
    data = json.load(f)

output_dir = project_dir / "data"
os.makedirs(output_dir, exist_ok=True)


# voice_programs = [52, 53, 54, 52]
voice_programs = [52, 52, 53, 53]
voice_names = ['Soprano', 'Alto', 'Tenor', 'Bass']

base_velocity = 80
voice_velocity_offsets = [0, -3, -2, -5]
voice_humanization = [2, 3, 2, 4]
sequences = data["train"]

sequences_to_midi(sequences, output_dir)

