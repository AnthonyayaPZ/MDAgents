import os
import json
import random
import argparse
from tqdm import tqdm
from termcolor import cprint
from utils_AD import (
    Agent, Group, parse_hierarchy, parse_group_info, setup_model,
    build_clinical_input, determine_difficulty,
    process_basic_query, process_intermediate_query, process_advanced_query
)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(
    description="MDAgents — Alzheimer's Disease Treatment Plan Generator")
parser.add_argument(
    '--text_report', type=str, required=True,
    help='Path to the clinical text report file (plain text)')
parser.add_argument(
    '--img_path', type=str, default=None,
    help='Path to the brain MRI image file (png/jpg)')
parser.add_argument(
    '--model', type=str, default='gpt-4o-mini',
    help='LLM model to use (default: gpt-4o-mini)')
parser.add_argument(
    '--difficulty', type=str, default='adaptive',
    choices=['basic', 'intermediate', 'advanced', 'adaptive'],
    help='Case complexity routing (default: adaptive)')
parser.add_argument(
    '--output', type=str, default=None,
    help='Output JSON file path (default: output/ad_treatment_plan.json)')
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
model, client = setup_model(args.model)

# ---------------------------------------------------------------------------
# Load clinical input
# ---------------------------------------------------------------------------
if not os.path.isfile(args.text_report):
    raise FileNotFoundError(
        f"Clinical text report not found: {args.text_report}")

with open(args.text_report, 'r') as f:
    text_report = f.read().strip()

if args.img_path and not os.path.isfile(args.img_path):
    raise FileNotFoundError(f"MRI image file not found: {args.img_path}")

clinical_input = build_clinical_input(text_report, args.img_path)

print("=" * 60)
cprint("MDAgents — Alzheimer's Disease Treatment Plan Generator",
       'cyan', attrs=['bold'])
print("=" * 60)
print(f"Model       : {args.model}")
print(f"Text report : {args.text_report}")
print(f"MRI image   : {args.img_path or 'Not provided'}")
print(f"Difficulty  : {args.difficulty}")
print("=" * 60)

# ---------------------------------------------------------------------------
# Determine case complexity
# ---------------------------------------------------------------------------
difficulty = determine_difficulty(
    clinical_input, args.difficulty, img_path=args.img_path)
cprint(f"\n[INFO] Case complexity determined: {difficulty}\n",
       'green', attrs=['bold'])

# ---------------------------------------------------------------------------
# Route to appropriate processing pathway
# ---------------------------------------------------------------------------
if difficulty == 'basic':
    treatment_plan = process_basic_query(
        clinical_input, args.model, img_path=args.img_path)
elif difficulty == 'intermediate':
    treatment_plan = process_intermediate_query(
        clinical_input, args.model, img_path=args.img_path)
elif difficulty == 'advanced':
    treatment_plan = process_advanced_query(
        clinical_input, args.model, img_path=args.img_path)

# ---------------------------------------------------------------------------
# Post-process: extract the JSON from the model response
# ---------------------------------------------------------------------------
def extract_json(response):
    """Try to extract a valid JSON object from the model response string."""
    if isinstance(response, dict):
        # temp_responses returns {temperature: text}
        for _temp, text in response.items():
            return extract_json(text)

    text = response.strip()

    # Strip markdown code fences if present
    if '```json' in text:
        text = text.split('```json')[1]
    if '```' in text:
        text = text.split('```')[0]

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find the first { ... } block
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    # Return raw text wrapped in a JSON structure if parsing fails
    return {"raw_response": text}


treatment_plan_json = extract_json(treatment_plan)

# ---------------------------------------------------------------------------
# Save output
# ---------------------------------------------------------------------------
output_dir = os.path.join(os.getcwd(), 'output')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_path = args.output or os.path.join(output_dir, 'ad_treatment_plan.json')
# Ensure the output directory exists for custom paths too
os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(treatment_plan_json, f, indent=4)

print("\n" + "=" * 60)
cprint(f"Treatment plan saved to: {output_path}", 'green', attrs=['bold'])
print("=" * 60)

# Pretty-print the plan to console
print(json.dumps(treatment_plan_json, indent=2))
