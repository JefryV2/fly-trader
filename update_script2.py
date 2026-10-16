#!/usr/bin/env python3
import re

# Read the script.js file
with open('web-app/script.js', 'r', encoding='utf-8', errors='replace') as f:
    content = f.read()

# Add a function to update neuron count display
updateFunction = '''function updateNeuronCountDisplay() {
  // Update elements that use data-neuron-count attribute
  document.querySelectorAll('[data-neuron-count]').forEach(el => {
    const originalText = el.textContent || el.innerText;
    const match = originalText.match(/\\$\\{N_NEURONS\\}/);
    if (match) {
      // Replace the placeholder with actual N_NEURONS value
      const newText = originalText.replace(match[0], N_NEURONS);
      if (el.textContent !== undefined) {
        el.textContent = newText;
      } else {
        el.innerText = newText;
      }
    }
  });
  
  // Update brain badge to include N_NEURONS
  if (qBrainBadge) {
    qBrainBadge.textContent = `🧠 Gen ${Math.floor(totalEpisodes / 25) + 1} · Ep ${totalEpisodes} · ${N_NEURONS}-LIF Neurons`;
  }
}

// Call this function on initial load and when N_NEURONS changes
'''

# Insert the function after the initTabs function
initTabsInsertPoint = content.find('function initTabs() {')
if (initTabsInsertPoint !== -1:
    # Find the end of initTabs function
    braceCount = 0
    pos = initTabsInsertPoint
    while pos < len(content) and braceCount >= 0:
        if content[pos] == '{':
            braceCount++
        elif content[pos] == '}':
            braceCount--
        pos += 1
    
    insertPos = pos
    # Insert the new function after initTabs
    content = content[:insertPos] + '\n\n' + updateFunction + '\n' + content[insertPos:]

# Also add call to updateNeuronCountDisplay in key places
# In handleStepResult function
content = content.replace(
    'drawOscilloscope();',
    'drawOscilloscope();\\n  updateNeuronCountDisplay();'
)

# In apiReset function  
content = content.replace(
    'updateBrainViz(spikeData, 0.0);',
    'updateBrainViz(spikeData, 0.0);\\n  updateNeuronCountDisplay();'
)

# Write the updated content back
with open('web-app/script.js', 'w', encoding='utf-8') as f:
    f.write(content)

print('Updated script.js with neuron count display function')
