import pandas as pd
import uuid
import os
import yaml

script_dir = os.path.dirname(os.path.abspath(__file__))

base_path = os.path.join(script_dir)
# Load config 
cfg_path = os.path.join(base_path, "config_eval.yaml")
output_dir = os.path.join(base_path, "testsets")
output_path = os.path.join(output_dir, "hotel_base_validation.parquet")

base_asin = "111418"
hotel_name = "Hilton Boston Park Plaza"  # <--- Define the name here
file_hash = str(uuid.uuid4())[:8]

if os.path.exists(cfg_path):
    import yaml as _yaml
    with open(cfg_path) as fh:
        cfg = _yaml.safe_load(fh)
        base_asin = cfg.get("testset", {}).get("base_parent_asin", base_asin)

# COMPARISON QUERIES  
comparison = [
    f"How does the location of {hotel_name} compare to nearby hotels?",
    f"Is {hotel_name} rated higher than Hilton Boston Downtown?",
    f"Compare the star rating of {hotel_name} with competitors.",
    f"Does {hotel_name} have more rooms than a boutique hotel?",
    f"Which hotel offers better views: {hotel_name} or the Sheraton?"
]

# REVIEW QUERIES 
review = [
    f"Do reviews for {hotel_name} mention noise complaints?",
    f"Are there repeated complaints about cleanliness at {hotel_name}?",
    f"Do guests praise the staff at {hotel_name}?",
    f"What do reviews say about the beds at {hotel_name}?",
    f"Are people satisfied with the room service at {hotel_name}?"
]

# EDGE CASES  
edge = [
    f"What is the exact address of {hotel_name}?",
    f"How many stars does {hotel_name} have?",
    f"When was {hotel_name} built or renovated?",
    f"Does {hotel_name} have a restaurant or dining onsite?",
    f"What is the phone number for {hotel_name}?"
]

rows = []
for i, q in enumerate(comparison + review + edge):
    rows.append({
        "question": q,
        "file_hash": file_hash,
        "parent_asin": base_asin,
        "evolution_type": "comparison" if i < 5 else ("review" if i < 10 else "edge")
    })

os.makedirs(output_dir, exist_ok=True)
df = pd.DataFrame(rows)
df.to_parquet(output_path, index=False)
print(f"Wrote validation queries to {output_path}")