document.addEventListener('DOMContentLoaded', () => {
  initCharts();
});

function initCharts() {
  const chartContainers = document.querySelectorAll('[data-chart]');
  chartContainers.forEach(container => {
    const type = container.dataset.chart;
    if (type === 'compression') {
      renderCompressionChart(container);
    }
  });
}

function renderCompressionChart(container) {
  const data = [
    { label: 'FP32', bits: 32, storage: 108 },
    { label: 'FP16', bits: 16, storage: 54 },
    { label: 'INT8', bits: 8.125, storage: 27.4 },
    { label: 'INT4', bits: 4.125, storage: 13.9 },
    { label: 'Binary', bits: 1.125, storage: 3.8 },
    { label: 'Ternary', bits: 1.7, storage: 5.8 },
    { label: 'Codebook', bits: 2.0, storage: 6.8 }
  ];

  const maxStorage = Math.max(...data.map(d => d.storage));

  container.innerHTML = data.map(d => {
    const percentage = (d.storage / maxStorage) * 100;
    return `
      <div class="compression-bar-row">
        <span class="compression-bar-label">${d.label}</span>
        <div class="compression-bar-container">
          <div class="compression-bar-fill" style="width: ${percentage}%;"></div>
        </div>
        <span class="compression-bar-value">~${d.storage.toFixed(1)} GB</span>
      </div>
    `;
  }).join('');
}

function createProgressBar(container, value, max, color = 'var(--accent-primary)') {
  const percentage = (value / max) * 100;
  container.innerHTML = `
    <div class="progress-bar">
      <div class="progress-bar-fill" style="width: ${percentage}%; background: ${color};"></div>
    </div>
  `;
}

function createTerminalLog(container, logs) {
  container.innerHTML = logs.map(log => {
    const type = log.type || 'output';
    const prompt = log.prompt || '';
    const message = log.message || '';
    return `<div class="terminal-line"><span class="terminal-${type}">${prompt}${message}</span></div>`;
  }).join('');
}