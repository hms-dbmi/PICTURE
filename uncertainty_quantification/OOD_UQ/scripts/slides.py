# Build a list of all the data. Each data point consist in the path of the slide, the outcome and the case_id.
# Example:  python scripts/slides.py --source='outcomes' --target='slides'

import json
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--source", help="Name of the JSON of the result of 'outcome.py' that should be in data folder, e.g 'outcomes'", type=str, default='outcomes')
parser.add_argument("--target", help="Target filename that will be written in data, e.g 'slides'",type=str, default='slides')
args = parser.parse_args()

source = args.source
target = args.target
  
def build_slides_list(source = 'outcomes', target = 'slides'):
  with open(f'data/{source}.json') as handle:
    outcomes = json.loads(handle.read())

  files = []
  for path in glob.glob('*/*/*.svs'):
    file = {'case_id': path.split('/')[2][:12], 'path': path}
    file['outcome'] = outcomes[file['case_id']]['vital_status']
    files.append(file)

  with open(f'data/{target}.json', 'w') as f:
    json.dump(files, f,indent='\t')

if __name__ == "__main__":
  build_slides_list(source,target)