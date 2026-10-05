document.addEventListener('DOMContentLoaded', () => {
  initCompressionPage();
});

function initCompressionPage() {
  const modelSizeSelect = document.getElementById('compression-model-size');
  if (!modelSizeSelect) return;

  const representations = [
    { key: 'fp32', bits: 32, label: 'FP32', color: '#6d737a' },
    { key: 'fp16', bits: 16, label: 'FP16', color: '#00d4ff' },
    { key: 'int8', bits: 8.125, label: 'INT8', color: '#00ff88' },
    { key: 'int4', bits: 4.125, label: 'INT4', color: '#ffb800' },
    { key: 'binary', bits: 1.125, label: 'Binary', color: '#ff4757' },
    { key: 'ternary', bits: 1.7, label: 'Ternary', color: '#b873ff' },
    { key: 'codebook', bits: 2.0, label: 'Codebook', color: '#00d4ff' }
  ];

  function updateBars() {
    const paramsB = parseFloat(modelSizeSelect.value.replace('B', ''));
    const maxStorage = (paramsB * 32) / 8;

    representations.forEach(rep => {
      const storage = (paramsB * rep.bits) / 8;
      const percentage = (storage / maxStorage) * 100;

      const bar = document.getElementById(`bar-${rep.key}`);
      const value = document.getElementById(`value-${rep.key}`);

      if (bar) {
        bar.style.width = `${Math.max(percentage, 2)}%`;
        bar.style.background = `linear-gradient(90deg, ${rep.color}, ${rep.color}88)`;
      }
      if (value) value.textContent = `~${storage.toFixed(1)} GB`;
    });
  }

  modelSizeSelect.addEventListener('change', updateBars);
  updateBars();
}