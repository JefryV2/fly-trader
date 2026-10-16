#!/usr/bin/env python3
import re

# Read the script.js file
with open('web-app/script.js', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# Make targeted replacements
# 1. Update the initial spikeData1024 declaration
content = content.replace(
    'let spikeData1024    = new Array(1024).fill(0);',
    'let spikeData = new Array(N_NEURONS).fill(0);'
)

# 2. Update the reset of spikeData in apiReset function
content = content.replace(
    'spikeData1024         = new Array(1024).fill(0);',
    'spikeData = new Array(N_NEURONS).fill(0);'
)

# 3. Update the assignment in handleStepResult function
content = content.replace(
    'spikeData1024   = d.spikes_full || new Array(1024).fill(0);',
    'spikeData = d.spikes_full || new Array(N_NEURONS).fill(0);'
)

# 4. Update drawHeatmap and related function calls
content = content.replace(
    'drawHeatmap(spikeData1024);',
    'drawHeatmap(spikeData);'
)
content = content.replace(
    'updateActivityBars(spikeData1024, null);',
    'updateActivityBars(spikeData, null);'
)
content = content.replace(
    'updateActivityBars(spikeData1024, d.regions);',
    'updateActivityBars(spikeData, d.regions);'
)

# 5. Update the brain badge initialization
content = content.replace(
    'const N_NEURONS = 1024;',
    'const N_NEURONS = 1024;'  # This one stays for now, but we'll update later
)

# 6 in updateBrainViz function
content = content.replace(
    'function updateBrainViz(spikes1024, dopamine) {',
    'function updateBrainViz(spikes, dopamine) {'
)
content = content.replace(
    'for (let i = 0; i < 1024; i++) {',
    'for (let i = 0; i < N_NEURONS; i++) {'
)

# 7. Update summary display in drawEvolutionChart
content = content.replace(
    'summaryDiv.innerHTML = `<span class="a-total">${totalSpikes} <span style="color:var(--txt-dim)">/ 1024 neurons active</span></span>`;',
    'summaryDiv.innerHTML = `<span class="a-total">${totalSpikes} <span style="color:var(--txt-dim)">/ ${N_NEURONS} neurons active</span></span>`;'
)

# Write the updated content back
with open('web-app/script.js', 'w', encoding='utf-8') as f:
    f.write(content)

print('Updated script.js with dynamic N_NEURONS')
