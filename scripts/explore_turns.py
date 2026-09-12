import json
import os

import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
MANIFEST_PATH = os.path.join(DATA_DIR, 'manifest.csv')
TURNS_DIR = os.path.join(DATA_DIR, 'turns')


def load_turns(anon_id):
    path = os.path.join(TURNS_DIR, f'{anon_id}.json')
    with open(path) as f:
        return json.load(f)['turns']


def caller_response_gaps(turns):
    """
    For each caller (channel 0) turn, find the gap since the
    previous agent (channel 1) turn ended. Returns a list of gaps
    in seconds.
    """
    gaps = []
    last_agent_end = None
    for turn in turns:
        if turn['channel'] == 1:
            last_agent_end = turn['end']
        elif turn['channel'] == 0 and last_agent_end is not None:
            gap = turn['start'] - last_agent_end
            if gap >= 0:
                gaps.append(gap)
            last_agent_end = None
    return gaps


def main():
    manifest = pd.read_csv(MANIFEST_PATH)

    rows = []
    for _, row in manifest.iterrows():
        try:
            turns = load_turns(row['anon_id'])
        except FileNotFoundError:
            continue
        gaps = caller_response_gaps(turns)
        if not gaps:
            continue
        rows.append({
            'anon_id': row['anon_id'],
            'label': row['label'],
            'split': row['split'],
            'mean_gap': sum(gaps) / len(gaps),
            'min_gap': min(gaps),
            'max_gap': max(gaps),
            'n_gaps': len(gaps),
        })

    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} calls\n")

    print("=== Mean response gap, by label ===")
    print(df.groupby('label')['mean_gap'].describe())

    print("\n=== Min response gap, by label ===")
    print(df.groupby('label')['min_gap'].describe())


if __name__ == '__main__':
    main()