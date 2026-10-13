const textInput = document.getElementById('text-input');
const analyzeBtn = document.getElementById('analyze-btn');
const clearBtn = document.getElementById('clear-btn');
const resultsSection = document.getElementById('results');
const traitGrid = document.getElementById('trait-grid');
const summarySection = document.getElementById('summary');
const summaryText = document.getElementById('summary-text');
const errorSection = document.getElementById('error');

const apiBaseUrl = 'http://127.0.0.1:8011';
const traitIcons = {
  agreeableness: '♡',
  openness: '▤',
  conscientiousness: '☑',
  extraversion: '◈',
  neuroticism: '☁',
};

function showError(message) {
  errorSection.textContent = message;
  errorSection.classList.remove('hidden');
}

function clearError() {
  errorSection.textContent = '';
  errorSection.classList.add('hidden');
}

function renderTraits(traits) {
  traitGrid.innerHTML = '';
  for (const [trait, value] of Object.entries(traits)) {
    const item = document.createElement('div');
    item.className = 'trait-item';
    const score = Math.max(0, Math.min(100, Number(value)));
    item.innerHTML = `
      <div class="trait-heading">
        <span class="trait-icon" aria-hidden="true">${traitIcons[trait] || '●'}</span>
        <span class="trait-name">${trait.toUpperCase()}</span>
        <span class="trait-score">Score: ${score.toFixed(0)}</span>
      </div>
      <div class="score-track" role="progressbar" aria-label="${trait}" aria-valuemin="0" aria-valuemax="100" aria-valuenow="${score.toFixed(0)}">
        <div class="score-fill" style="width: ${score}%"></div>
      </div>
      <div class="score-scale"><span>0</span><span>20</span><span>40</span><span>60</span><span>80</span><span>100</span></div>
    `;
    traitGrid.appendChild(item);
  }
}

function renderSummary(summary) {
  summaryText.textContent = summary;
  summarySection.classList.remove('hidden');
}

async function analyzeText() {
  const text = textInput.value.trim();
  if (!text) {
    showError('Please enter some text before analyzing.');
    return;
  }

  clearError();
  resultsSection.classList.remove('hidden');
  analyzeBtn.disabled = true;
  analyzeBtn.textContent = 'Analyzing...';

  try {
    const request = {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text }),
    };
    const [predictionResponse, summaryResponse] = await Promise.all([
      fetch(`${apiBaseUrl}/predict`, request),
      fetch(`${apiBaseUrl}/summarize`, request),
    ]);

    const predictionPayload = await predictionResponse.json();
    const summaryPayload = await summaryResponse.json();
    if (!predictionResponse.ok) {
      throw new Error(predictionPayload.detail || 'Prediction failed');
    }
    if (!summaryResponse.ok) {
      throw new Error(summaryPayload.detail || 'Summary failed');
    }

    renderTraits(predictionPayload.traits);
    renderSummary(summaryPayload.summary);
  } catch (error) {
    showError(`Request failed: ${error.message}. Ensure the API server is running at ${apiBaseUrl}.`);
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = 'Analyze';
  }
}

clearBtn.addEventListener('click', () => {
  textInput.value = '';
  clearError();
  resultsSection.classList.add('hidden');
  summarySection.classList.add('hidden');
  summaryText.textContent = '';
  textInput.focus();
});

analyzeBtn.addEventListener('click', analyzeText);
