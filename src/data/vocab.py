# This script containins utilities to encode the pitch values to a continuous natural set (0 is padding token, 1 is rest token, and from 2 to


PAD_PITCH = 0
REST_PITCH = -1

def encode_pitch(pitch: int, min_pitch: int):
    if pitch == REST_PITCH:
        return 1
    elif pitch == PAD_PITCH:
        return PAD_PITCH
    else:
        return pitch - min_pitch + 2


def get_vocab_size(min_pitch: int, max_pitch: int):
    return max_pitch - min_pitch + 3