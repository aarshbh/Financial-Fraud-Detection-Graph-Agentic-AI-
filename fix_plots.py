import os

files = [
    'part2_graph_features.py',
    'part3_fraud_detection.py',
    'part5_results_visualization.py',
]
base = r'D:\New\2026\Subjects\Pattern Recognition\Fraud_Detection_In_Financial_Banking_System'

agg_lines = 'import matplotlib\nmatplotlib.use("Agg")  # non-interactive: save to file\n'

for fn in files:
    path = os.path.join(base, fn)
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    if 'matplotlib.use' not in content:
        content = content.replace(
            'import matplotlib.pyplot as plt',
            agg_lines + 'import matplotlib.pyplot as plt',
            1
        )
    content = content.replace('    plt.show()', '    plt.close()')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'Fixed: {fn}')
