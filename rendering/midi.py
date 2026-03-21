import random
from pathlib import Path
import pretty_midi


def sequences_to_midi(
        sequences: list,
        outputs_dir: Path,
        timestamp: str = None,
        voice_programs= [52, 52, 53, 53],
        base_velocity=85,
        voice_velocity_offsets=[5, -5, 0, 10],
        voice_humanization=[3, 2, 3, 2],
        tempo_bpm=72,
        return_output_paths: bool = False,
        ref_seq_len: int = None
):
    """
    Convert a batch of sequence tensors to MIDI files.
    """
    print(f"Using tempo_bpm: {tempo_bpm}")
    voice_names = ["Soprano", "Alto", "Tenor", "Bass"]
    time_step = 60 / tempo_bpm / 4  # sixteenth note duration in seconds
    print(f"Time step (per token): {time_step}")


    output_paths = []
    for idx, sequence in enumerate(sequences):
        # Initialize PrettyMIDI
        expected_duration = time_step * len(sequence)
        print(f"Actual sequence Length: {len(sequence)}")
        print(f"Expected duration: {expected_duration:.2f} seconds")
        pm = pretty_midi.PrettyMIDI(initial_tempo=tempo_bpm)
        instruments = [
            pretty_midi.Instrument(program=program, name=voice_names[i])
            for i, program in enumerate(voice_programs)
        ]

        # Tracking previous pitches to sustain notes
        prev_pitches = [-1] * len(voice_programs)
        note_starts = [0.0] * len(voice_programs)
        prev_velocities = [base_velocity] * len(voice_programs)

        start_time = 0.0

        for timestep in sequence:
            for voice_idx, pitch in enumerate(timestep):
                prev_pitch = prev_pitches[voice_idx]

                if pitch != prev_pitch:
                    if prev_pitch != -1:
                        note = pretty_midi.Note(
                            velocity=int(prev_velocities[voice_idx]),
                            pitch=int(prev_pitch),
                            start=note_starts[voice_idx],
                            end=start_time
                        )
                        instruments[voice_idx].notes.append(note)

                    if pitch != -1 and 0 <= pitch <= 127:
                        velocity = base_velocity + voice_velocity_offsets[voice_idx]
                        velocity += random.randint(
                            -voice_humanization[voice_idx],
                            voice_humanization[voice_idx]
                        )
                        velocity = max(40, min(velocity, 127))

                        note_starts[voice_idx] = start_time
                        prev_velocities[voice_idx] = velocity

                prev_pitches[voice_idx] = pitch

            start_time += time_step

        for voice_idx, pitch in enumerate(prev_pitches):
            if pitch != -1:
                note = pretty_midi.Note(
                    velocity=int(prev_velocities[voice_idx]),
                    pitch=int(pitch),
                    start=note_starts[voice_idx],
                    end=start_time
                )
                instruments[voice_idx].notes.append(note)

        for instrument in instruments:
            pm.instruments.append(instrument)

        midi_dir = outputs_dir / "midi"
        midi_dir.mkdir(parents=True, exist_ok=True)

        if timestamp is not None:
            output_path = midi_dir / f"{timestamp}_sample{idx + 1}.mid"

        else:
            output_path = midi_dir / f"sample{idx + 1}.mid"

        output_paths.append(output_path)
        pm.write(output_path)

        if return_output_paths:
            return output_paths

