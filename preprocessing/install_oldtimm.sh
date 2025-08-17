#!/bin/bash
# installs the deprecated timm library as `oldtimm`

# === setup ===
mkdir .oldtimm
tar -xvf ./timm-0.5.4.tar -C .oldtimm
cd .oldtimm
mv timm-0.5.4 oldtimm
cd oldtimm
mv timm oldtimm
mv timm.egg-info oldtimm.egg-info

# replace only import statements
find . -type f -name '*.py' -exec perl -pi -e 's/^(\s*(from|import)\s+)timm\b/${1}oldtimm/g' {} \;

# replace module prefixes
find . -type f -name '*.py' -exec perl -pi -e 's/\btimm\./oldtimm./g' {} \;

# replace top-level issues
perl -pi -e 's/\btimm\b/oldtimm/g' setup.py
find oldtimm.egg-info -type f -exec perl -pi -e 's/\btimm\b/oldtimm/g' {} \;
find . -maxdepth 1 -type f -exec perl -pi -e 's/\btimm\b/oldtimm/g' {} \;

# install the timm-0.5.4 library as oldtimm
pip install -e .

