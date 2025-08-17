# Build a dictionary with key the CASE_ID and value the bio description of the patient (Sex, Outcome etc.)
# Example:  python scripts/outcomes.py --source='clinical.cases_selection.2022-08-11.json'

import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--source", help="Name of te JSON of the clinical case selection that should be in the data folder, e.g 'clinical'", type=str)
parser.add_argument(
    "--target", help="Target filename that will be written in data, e.g 'outcomes'", type=str, default='outcomes')
args = parser.parse_args()

source = args.source
target = args.target

def build_dict_outcomes(source='clinical', target='outcomes'):
    with open(f'data/{source}') as handle:
        dictdump = json.loads(handle.read())

    caseid_to_outcome = {i['submitter_id']: i['demographic'] for i in dictdump}

    with open(f'data/{target}.json', 'w') as f:
        json.dump(caseid_to_outcome, f, indent='\t')

if __name__ == "__main__":
    build_dict_outcomes(source,target)
