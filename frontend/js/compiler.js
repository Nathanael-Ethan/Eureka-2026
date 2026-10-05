document.addEventListener('DOMContentLoaded', () => {
  initCompilerPage();
});

function initCompilerPage() {
  const form = document.getElementById('compiler-form');
  if (!form) return;

  const analyzeBtn = document.getElementById('analyze-btn');
  const resultsDiv = document.getElementById('compiler-results');
  const logDiv = document.getElementById('compiler-log');

  analyzeBtn.addEventListener('click', () => {
    const formData = new FormData(form);
    const modelSize = formData.get('model-size');
    const originalPrecision = formData.get('original-precision');
    const targetRep = formData.get('target-representation');
    const hardware = formData.get('hardware');
    const ram = formData.get('ram');
    const storage = formData.get('storage');
    const contextLength = formData.get('context-length');

    const paramsB = parseFloat(modelSize.replace('B', ''));
    const params = paramsB * 1e9;

    const precisionBits = { FP32: 32, FP16: 16, BF16: 16 };
    const targetBits = { FP16: 16, INT8: 8.125, INT4: 4.125, Binary: 1.125, Ternary: 1.7, Codebook: 2.0 };

    const originalBits = precisionBits[originalPrecision] || 16;
    const compressedBits = targetBits[targetRep] || 4.125;

    const originalStorage = (params * originalBits) / 8 / 1e9;
    const compressedStorage = (params * compressedBits) / 8 / 1e9;
    const compressionRatio = originalStorage / compressedStorage;

    const runtimeMemory = compressedStorage * 1.2;
    const kvCache = (parseFloat(contextLength) * 0.5 * paramsB * 2) / 1e9;

    const storageFits = compressedStorage <= parseFloat(storage);
    const ramFits = runtimeMemory <= parseFloat(ram);

    const logs = [
      `[LOAD] Loading model: ${paramsB}B parameters`,
      `[LOAD] Original precision: ${originalPrecision} (${originalBits} bits)`,
      `[ANALYZE] Parameter count: ${paramsB}B`,
      `[ANALYZE] Tensor count: ~${Math.round(paramsB * 2)}`,
      `[ANALYZE] Original storage: ~${originalStorage.toFixed(1)} GB`,
      `[PLAN] Target representation: ${targetRep} (${compressedBits} bits)`,
      `[PLAN] Hardware: ${hardware}, ${ram}GB RAM, ${storage}GB storage`,
      `[PLAN] Context length: ${contextLength}`,
      `[TRANSFORM] Applying ${targetRep} quantization...`,
      `[TRANSFORM] Compression ratio: ${compressionRatio.toFixed(1)}x`,
      `[VALIDATE] Reconstruction error: pending measurement`,
      `[VALIDATE] Storage: ~${compressedStorage.toFixed(1)} GB`,
      `[VALIDATE] Runtime memory: ~${runtimeMemory.toFixed(1)} GB`,
      `[EXPORT] Creating LDMARK artifact...`,
      `[RUNTIME] Estimated KV-cache: ~${kvCache.toFixed(2)} GB`,
      `[DONE] Analysis complete.`
    ];

    if (logDiv) {
      logDiv.innerHTML = logs.map(line => {
        const type = line.startsWith('[LOAD]') ? 'terminal-info' :
                     line.startsWith('[ANALYZE]') ? 'terminal-info' :
                     line.startsWith('[PLAN]') ? 'terminal-info' :
                     line.startsWith('[TRANSFORM]') ? 'terminal-warning' :
                     line.startsWith('[VALIDATE]') ? 'terminal-success' :
                     line.startsWith('[EXPORT]') ? 'terminal-info' :
                     line.startsWith('[DONE]') ? 'terminal-success' : 'terminal-output';
        return `<div class="terminal-line"><span class="${type}">${line}</span></div>`;
      }).join('');
    }

    document.getElementById('result-params').textContent = `${paramsB}B`;
    document.getElementById('result-original-storage').textContent = `~${originalStorage.toFixed(1)} GB`;
    document.getElementById('result-compressed-storage').textContent = `~${compressedStorage.toFixed(1)} GB`;
    document.getElementById('result-compression-ratio').textContent = `${compressionRatio.toFixed(1)}x`;
    document.getElementById('result-runtime-memory').textContent = `~${runtimeMemory.toFixed(1)} GB`;
    document.getElementById('result-kv-cache').textContent = `~${kvCache.toFixed(2)} GB`;

    const storageFitsEl = document.getElementById('result-storage-fits');
    storageFitsEl.textContent = storageFits ? 'YES' : 'NO';
    storageFitsEl.className = `result-value ${storageFits ? 'success' : 'danger'}`;

    const compatEl = document.getElementById('result-hardware-compat');
    if (storageFits && ramFits) {
      compatEl.textContent = 'Compatible';
      compatEl.className = 'result-value success';
    } else if (storageFits && !ramFits) {
      compatEl.textContent = 'Storage OK, RAM insufficient';
      compatEl.className = 'result-value warning';
    } else {
      compatEl.textContent = 'Incompatible';
      compatEl.className = 'result-value danger';
    }

    resultsDiv.style.display = 'block';
    resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  });
}