from src.data.dataset import BachDataset

try:
    ds = BachDataset(split='train')
    print(f"MIN_PITCH: {ds.get_min_pitch()}")
except Exception as e:
    print(f"Error: {e}")
