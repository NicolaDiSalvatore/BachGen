"""
This script is made to convert each MIDI chorale file into a WAV file.
"""

import os
import subprocess

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
soundfont_path = os.path.join(project_dir,  "resources", "052_Florestan_Ahh_Choir.sf2")
midi_dir = os.path.join(project_dir, "data", "raw", "audio_midi")
output_dir = os.path.join(project_dir, "data", "raw", "audio_wav")
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(midi_dir):
    if filename.endswith(".mid"):
        midi_path = os.path.join(midi_dir, filename)
        wav_path = os.path.join(output_dir, filename.replace(".mid", ".wav"))

        command = [
            "fluidsynth",
            "-F", wav_path,
            "-r", "44100",
            "-ni",
            soundfont_path,
            midi_path
        ]

        print(f"Rendering {filename}...")
        subprocess.run(command, check=True)

print("✅ All MIDI files rendered to WAV.")