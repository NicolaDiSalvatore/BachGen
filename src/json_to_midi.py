"""
This script is made to convert each chorale into a MIDI file.
"""

import json
import pretty_midi
import os
import random

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
json_file = os.path.join(project_dir, "data", "raw", "Jsb16thSeparated.json")
with open(json_file, 'r') as f:
    data = json.load(f)


output_dir = os.path.join(project_dir, "data", "raw", "audio_midi")
os.makedirs(output_dir, exist_ok=True)

# Define voice programs (General MIDI: 0-based)
voice_programs = [52, 53, 54, 52]  # S, A, T, B
voice_names = ['Soprano', 'Alto', 'Tenor', 'Bass']

# Velocity settings for natural choral balance
base_velocity = 80 
voice_velocity_offsets = [0, -3, -2, -5]  # S > T > A > B
voice_humanization = [2, 3, 2, 4]  # More variation in lower voices

# Process each dataset split (train, test, valid)
for split_name, chorales in data.items():
    print(f"Processing {split_name} set...")
    
    for chorale_idx, chorale in enumerate(chorales):
        # Initialize PrettyMIDI object
        pm = pretty_midi.PrettyMIDI()
        instruments = [
            pretty_midi.Instrument(program=program, name=voice_names[i])
            for i, program in enumerate(voice_programs)
        ]

        # Constants
        time_step = 0.3  # Sixteenth note duration
        start_time = 0.0  # Reset for each chorale

        for timestep in chorale:
            for voice_idx, pitch in enumerate(timestep):
                if pitch != -1:
                    duration = time_step

                    # Calculate velocity with natural offset and voice-specific humanization
                    velocity = base_velocity + voice_velocity_offsets[voice_idx]
                    velocity += random.randint(-voice_humanization[voice_idx], 
                                            voice_humanization[voice_idx])
                    velocity = max(40, min(velocity, 127))  # Clamp to valid range

                    note = pretty_midi.Note(
                        velocity=velocity,
                        pitch=pitch,
                        start=start_time,
                        end=start_time + duration
                    )
                    instruments[voice_idx].notes.append(note)
            start_time += time_step

        # Add all voice instruments to the MIDI object
        for instrument in instruments:
            pm.instruments.append(instrument)

        # Output filename
        output_filename = f"{split_name}_chorale_{chorale_idx:03d}.mid"
        output_path = os.path.join(output_dir, output_filename)
        pm.write(output_path)

        if (chorale_idx + 1) % 10 == 0:
            print(f"Processed {chorale_idx + 1} chorales in {split_name} set")

print(f"\nAll MIDI files have been created in the '{output_dir}' directory.")