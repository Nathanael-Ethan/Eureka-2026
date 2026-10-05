document.addEventListener('DOMContentLoaded', () => {
  initNavigation();
  initHeroPipeline();
  initCompilerDemo();
  initCompressionViz();
  initPipelineStages();
  initModals();
});

function initNavigation() {
  const toggle = document.querySelector('.navbar-toggle');
  const nav = document.querySelector('.navbar-nav');

  if (toggle && nav) {
    toggle.addEventListener('click', () => {
      const isOpen = nav.classList.contains('open');
      nav.classList.toggle('open');
      toggle.setAttribute('aria-expanded', !isOpen);
    });
  }

  const currentPage = window.location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('.navbar-link').forEach(link => {
    const href = link.getAttribute('href');
    if (href === currentPage || (currentPage === '' && href === 'index.html')) {
      link.classList.add('active');
    }
  });
}

function initHeroPipeline() {
  const stages = document.querySelectorAll('#hero-pipeline .pipeline-stage');
  if (stages.length === 0) return;

  let currentIndex = 0;

  function activateStage(index) {
    stages.forEach((stage, i) => {
      stage.classList.toggle('active', i === index);
    });
  }

  function nextStage() {
    currentIndex = (currentIndex + 1) % stages.length;
    activateStage(currentIndex);
  }

  activateStage(0);
  setInterval(nextStage, 1500);
}

function initCompilerDemo() {
  const analyzeBtn = document.getElementById('analyze-btn');
  const resultsDiv = document.getElementById('compiler-results');
  const modelSizeSelect = document.getElementById('model-size');
  const customParamsGroup = document.getElementById('custom-params-group');
  const customParamsInput = document.getElementById('custom-params');

  if (!analyzeBtn || !resultsDiv) return;

  if (modelSizeSelect && customParamsGroup) {
    modelSizeSelect.addEventListener('change', () => {
      customParamsGroup.style.display = modelSizeSelect.value === 'custom' ? 'block' : 'none';
    });
  }

  analyzeBtn.addEventListener('click', () => {
    const modelSize = modelSizeSelect ? modelSizeSelect.value : '27B';
    const customParams = customParamsInput ? parseFloat(customParamsInput.value) : 27;
    const originalPrecision = document.getElementById('original-precision')?.value || 'FP16';
    const targetRep = document.getElementById('target-representation')?.value || 'INT4';
    const hardwareType = document.getElementById('hardware-type')?.value || 'desktop-cpu';
    const ram = document.getElementById('ram')?.value || '16';
    const storageBudget = document.getElementById('storage-budget')?.value || '32';
    const contextLength = document.getElementById('context-length')?.value || '4096';

    const paramsB = modelSize === 'custom' ? customParams : parseFloat(modelSize.replace('B', ''));
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

    const storageFits = compressedStorage <= parseFloat(storageBudget);
    const ramFits = runtimeMemory <= parseFloat(ram);

    document.getElementById('result-params').textContent = `${paramsB}B`;
    document.getElementById('result-original-storage').textContent = `~${originalStorage.toFixed(1)} GB`;
    document.getElementById('result-compressed-storage').textContent = `~${compressedStorage.toFixed(1)} GB`;
    document.getElementById('result-compression-ratio').textContent = `${compressionRatio.toFixed(1)}×`;
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

function initCompressionViz() {
  const modelSizeSelect = document.getElementById('compression-model-size');
  if (!modelSizeSelect) return;

  const representations = [
    { key: 'fp32', bits: 32, label: 'FP32' },
    { key: 'fp16', bits: 16, label: 'FP16' },
    { key: 'int8', bits: 8.125, label: 'INT8' },
    { key: 'int4', bits: 4.125, label: 'INT4' },
    { key: 'binary', bits: 1.125, label: 'Binary' },
    { key: 'ternary', bits: 1.7, label: 'Ternary' },
    { key: 'codebook', bits: 2.0, label: 'Codebook' }
  ];

  function updateBars() {
    const paramsB = parseFloat(modelSizeSelect.value.replace('B', ''));
    const maxStorage = (paramsB * 32) / 8;

    representations.forEach(rep => {
      const storage = (paramsB * rep.bits) / 8;
      const percentage = (storage / maxStorage) * 100;

      const bar = document.getElementById(`bar-${rep.key}`);
      const value = document.getElementById(`value-${rep.key}`);

      if (bar) bar.style.width = `${Math.max(percentage, 2)}%`;
      if (value) value.textContent = `~${storage.toFixed(1)} GB`;
    });
  }

  modelSizeSelect.addEventListener('change', updateBars);
  updateBars();
}

function initPipelineStages() {
  const cards = document.querySelectorAll('.pipeline-stage-card');
  if (cards.length === 0) return;

  const stageInfo = {
    load: {
      title: '01 — LOAD',
      content: `
        <h4>Overview</h4>
        <p>The LOAD stage reads model files from disk and prepares them for analysis.</p>
        <h4>Supported Formats</h4>
        <ul>
          <li>PyTorch (.pt, .pth, .bin)</li>
          <li>Safetensors (.safetensors)</li>
          <li>NumPy (.npy, .npz)</li>
        </ul>
        <h4>Output</h4>
        <p>Raw tensor data ready for analysis.</p>
      `
    },
    analyze: {
      title: '02 — ANALYZE',
      content: `
        <h4>Overview</h4>
        <p>The ANALYZE stage examines the model structure and characteristics.</p>
        <h4>Examines</h4>
        <ul>
          <li>Parameter count</li>
          <li>Tensor count and shapes</li>
          <li>Data types (dtypes)</li>
          <li>Architecture detection</li>
          <li>Storage requirements</li>
          <li>Runtime memory estimates</li>
        </ul>
      `
    },
    plan: {
      title: '03 — PLAN',
      content: `
        <h4>Overview</h4>
        <p>The PLAN stage selects an optimal compression strategy.</p>
        <h4>Considers</h4>
        <ul>
          <li>Target hardware capabilities</li>
          <li>Storage budget constraints</li>
          <li>Runtime memory limits</li>
          <li>Context length requirements</li>
          <li>Available representations</li>
          <li>User quality requirements</li>
        </ul>
      `
    },
    transform: {
      title: '04 — TRANSFORM',
      content: `
        <h4>Overview</h4>
        <p>The TRANSFORM stage applies the selected compression representation.</p>
        <h4>Available Transformations</h4>
        <ul>
          <li>INT8 quantization</li>
          <li>INT4 quantization (group-wise)</li>
          <li>Binary quantization</li>
          <li>Ternary quantization</li>
          <li>Codebook-based representations</li>
        </ul>
      `
    },
    validate: {
      title: '05 — VALIDATE',
      content: `
        <h4>Overview</h4>
        <p>The VALIDATE stage verifies the quality of the transformation.</p>
        <h4>Checks</h4>
        <ul>
          <li>Reconstruction error (MAE, relative error)</li>
          <li>Storage size verification</li>
          <li>Tensor integrity</li>
          <li>Numerical consistency</li>
          <li>Metadata completeness</li>
        </ul>
      `
    },
    export: {
      title: '06 — EXPORT',
      content: `
        <h4>Overview</h4>
        <p>The EXPORT stage creates a versioned LDMARK artifact.</p>
        <h4>Artifact Contents</h4>
        <ul>
          <li>Compressed tensor data</li>
          <li>Model metadata</li>
          <li>Compression metadata</li>
          <li>Validation metadata</li>
          <li>Manifest and version info</li>
        </ul>
      `
    },
    runtime: {
      title: '07 — RUNTIME',
      content: `
        <h4>Overview</h4>
        <p>The RUNTIME stage loads and executes the compiled representation.</p>
        <h4>Capabilities</h4>
        <ul>
          <li>Load LDMARK artifacts</li>
          <li>Decompress tensors as needed</li>
          <li>Execute inference</li>
          <li>Report performance metrics</li>
        </ul>
      `
    }
  };

  cards.forEach(card => {
    card.addEventListener('click', () => {
      const stage = card.dataset.stage;
      const info = stageInfo[stage];
      if (info) {
        showModal(info.title, info.content);
      }
    });
  });
}

function initModals() {
  document.addEventListener('click', (e) => {
    if (e.target.classList.contains('modal-overlay')) {
      closeModal();
    }
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      closeModal();
    }
  });
}

function showModal(title, content) {
  closeModal();

  const overlay = document.createElement('div');
  overlay.className = 'modal-overlay';
  overlay.innerHTML = `
    <div class="modal" role="dialog" aria-modal="true" aria-labelledby="modal-title">
      <div class="modal-header">
        <h3 class="modal-title" id="modal-title">${title}</h3>
        <button class="modal-close" aria-label="Close modal">&times;</button>
      </div>
      <div class="modal-body">${content}</div>
    </div>
  `;

  document.body.appendChild(overlay);

  overlay.querySelector('.modal-close').addEventListener('click', closeModal);
  overlay.addEventListener('click', (e) => {
    if (e.target === overlay) closeModal();
  });
}

function closeModal() {
  const existing = document.querySelector('.modal-overlay');
  if (existing) existing.remove();
}

function showToast(message, type = 'success') {
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span class="toast-message">${message}</span>`;
  document.body.appendChild(toast);

  setTimeout(() => {
    toast.remove();
  }, 3000);
}