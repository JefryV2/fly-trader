#!/usr/bin/env python3
import re

# Read the script.js file
with open('web-app/script.js', 'r', encoding='utf-8', errors='replace') as f:
    lines = f.readlines()

# Find the position of handleStepResult function and add updateNeuronCountDisplay
for i, line in enumerate(lines):
    if 'function handleStepResult(d) {' in line:
        # Find the end of handleStepResult function
        for j in range(i, len(lines)):
            if 'drawOscilloscope();' in lines[j]:
                # Insert updateNeuronCountDisplay after drawOscilloscope
                lines.insert(j + 1, '  updateNeuronCountDisplay();\\n')
                break
        break

# Find the position of apiReset function and add updateNeuronCountDisplay  
for i, line in enumerate(lines):
    if 'async function apiReset() {' in line:
        # Find the line with updateBrainViz(spikeData, 0.0);
        for j in range(i, len(lines)):
            if 'updateBrainViz(spikeData, 0.0);' in lines[j]:
                # Insert updateNeuronCountDisplay after updateBrainViz
                lines.insert(j + 1, '  updateNeuronCountDisplay();\\n')
                break
        break

# Add the updateNeuronCountDisplay function before initTabs
for i, line in enumerate(lines):
    if 'function initTabs() {' in line:
        # Insert before initTabs
        insertPos = i
        updateFunc = [
            'function updateNeuronCountDisplay() {\\n',
            '  // Update elements that use data-neuron-count attribute\\n',
            '  document.querySelectorAll(\'[data-neuron-count]\\').forEach(el => {\\n',
            '    const originalText = el.textContent || el.innerText;\\n',
            '    const match = originalText.match(/\\$\\\\{N_NEURONS\\}/);\\n',
            '    if (match) {\\n',
            '      // Replace the placeholder with actual N_NEURONS value\\n',
            '      const newText = originalText.replace(match[0], N_NEURONS);\\n',
            '      if (el.textContent !== undefined) {\\n',
            '        el.textContent = newText;\\n',
            '      } else {\\n',
            '        el.innerText = newText;\\n',
            '      }\\n',
            '    }\\n',
            '  });\\n',
            '  \\n',
            '  // Update brain badge to include N_NEURONS\\n',
            '  if (qBrainBadge) {\\n',
            '    qBrainBadge.textContent = `🧠 Gen ${Math.floor(totalEpisodes / 25) + 1} · Ep ${totalEpisodes} · ${N_NEURONS}-LIF Neurons`;\\n',
            '  }\\n',
            '}\\n',
            '\\n'
        ]
        lines[insertPos:insertPos] = updateFunc
        break

# Write the updated content back
with open('web-app/script.js', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print('Updated script.js with neuron count display function')
