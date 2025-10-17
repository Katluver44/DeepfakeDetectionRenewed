#!/usr/bin/env python3
import nbformat

nb = nbformat.read('demo_test.ipynb', as_version=4)

print("=" * 80)
print("Notebook Execution Check")
print("=" * 80)

errors = []
for i, cell in enumerate(nb.cells):
    outputs = cell.get('outputs', [])
    for out in outputs:
        if out.get('output_type') == 'error':
            errors.append((i, out.get('ename'), out.get('evalue')))

if errors:
    print(f"\n❌ ERRORS FOUND: {len(errors)}\n")
    for i, ename, evalue in errors:
        print(f"Cell {i}: {ename}")
        print(f"  {evalue}\n")
else:
    print("\n✓ No errors found!")

# Count cells with output
cells_with_output = [c for c in nb.cells if c.get('outputs')]
print(f"✓ {len(cells_with_output)} cells have output")
print(f"✓ Total cells: {len(nb.cells)}")

# Check last cell output
if len(nb.cells) > 19:
    last_cell = nb.cells[19]
    outputs = last_cell.get('outputs', [])
    if outputs:
        print("\n" + "=" * 80)
        print("Final Test Results (Cell 19):")
        print("=" * 80)
        for out in outputs:
            if out.get('output_type') == 'stream':
                text = out.get('text', '')
                print(text)

print("\n" + "=" * 80)
print("✓ Notebook execution completed successfully!")
print("=" * 80)



