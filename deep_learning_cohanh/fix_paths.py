import json

with open('/home/shin-bkb/code/AIO_homework/deep_learning_cohanh/exercise1b.ipynb', 'r') as f:
    nb = json.load(f)

# Cell 1: remove google colab drive mount
nb['cells'][1]['source'] = ['# Running locally - no drive mount needed\n']

# Cell 2: replace %cd colab path with local path
nb['cells'][2]['source'] = [
    "import os\n",
    "os.chdir('/home/shin-bkb/code/AIO_homework/deep_learning_cohanh')\n",
    "print('Working directory:', os.getcwd())\n"
]
# Clear old output
nb['cells'][2]['outputs'] = []

with open('/home/shin-bkb/code/AIO_homework/deep_learning_cohanh/exercise1b.ipynb', 'w') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print('Done! Cells 1 & 2 updated.')
print('\nCell 1:', ''.join(nb['cells'][1]['source']))
print('Cell 2:', ''.join(nb['cells'][2]['source']))
