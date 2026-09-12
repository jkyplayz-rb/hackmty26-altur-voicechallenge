# Team Roles & Responsibilities

## 1. Acoustic Detection Lead
**Primary modules:** `app/detection/acoustic/`

- Find and integrate a pretrained voice deepfake/anti-spoofing classifier
- Run it against caller audio (channel 0) and produce a confidence score
- Evaluate performance on the train/val split, tune thresholds
- Document what the model looks at and why it works
- Owns model evaluation scripts

---

## 2. Behavioral Signal Lead
**Primary modules:** `app/detection/behavioral/`

- Extract features from `turns.json` (response gaps, interruption
  handling, silence recovery patterns)
- Validate signal strength across the full dataset (not just the
  initial sample)
- Turn findings into a scorable feature set for the classifier
- Owns `scripts/explore_turns.py` and related analysis

---

## 3. Lead Developer — Integration & Deployment
**Primary modules:** `app/`, deployment config

- Own the Flask `/detect` endpoint end-to-end
- Combine acoustic + behavioral signals into one final verdict
- Review and merge all team branches
- Handle live deployment, ensure endpoint is reachable during judging
- Owns `app/main.py` and the overall repo

---

## 4. Documentation & Demo Lead
**Primary modules:** `README.md`, demo prep

- Write the approach explanation required by the challenge
- Prepare the walkthrough for the judges' 15-minute visit
- Test the `/detect` endpoint manually before judging (edge cases,
  malformed input, latency)
- Track what's done vs. pending so the team always knows status

---

## GitHub Project Board Columns
`Backlog` → `In Progress` → `In Review` → `Done`

## Issue Labels
`acoustic` | `behavioral` | `api` | `deployment` | `docs` | `bug` | `demo-ready`