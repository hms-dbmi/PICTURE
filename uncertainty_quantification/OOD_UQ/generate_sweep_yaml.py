# generate sweep yaml file for the sweep experiment
import yaml
import os

template_yaml = 'sweepCV_vienna_CTransFeature_template.yaml'
outfolder = 'generated_sweep_yamls'
os.makedirs(outfolder, exist_ok=True)
# read template_yaml
with open(template_yaml, 'r') as f:
    # read template yaml as text file
    template = f.read()

    # template = yaml.load(f, Loader=yaml.FullLoader)
    # convert to string
    # template = yaml.dump(template)
    
for fold in range(10):
    # replace placeholder keyword
    out_yaml = template.replace('[FOLD]', str(fold))
    # write to file
    out_yaml_name = f'sweepCV_vienna_CTransFeature_fold{fold}.yaml'
    with open(os.path.join(outfolder, out_yaml_name), 'w') as f:
        f.write(out_yaml)

## generate experiment yaml file for the experiment
exp_template_yaml = 'configs/experiment/CTransFeatures_vienna_w_benign_quick_template.yaml'
outfolder = os.path.dirname(exp_template_yaml)

# read template_yaml
with open(exp_template_yaml, 'r') as f:
    # template = yaml.load(f, Loader=yaml.FullLoader)
    # # convert to string
    # template = yaml.dump(template)
    template = f.read()

for fold in range(10):
    # replace placeholder keyword
    out_yaml = template.replace('[FOLD]', str(fold))
    # write to file
    out_yaml_name = os.path.basename(exp_template_yaml).replace('template', f'fold{fold}')
    with open(os.path.join(outfolder, out_yaml_name), 'w') as f:
        f.write(out_yaml)


    
    


