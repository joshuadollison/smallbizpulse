let latestCandidates = [];

const state = {
  selectedBusinessId: null,
};

document.addEventListener('DOMContentLoaded', () => {
  bindEvents();
  loadHealth();
});

function bindEvents() {
  const form = document.getElementById('search-form');
  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    await runSearch();
  });
}

async function loadHealth() {
  const healthPill = document.getElementById('health-pill');
  try {
    const health = await fetchJson('/api/health');
    const checks = health.checks || {};

    const mode = checks.live_fallback_ready
      ? 'Artifact + live fallback ready'
      : 'Artifact-first mode';

    healthPill.textContent = mode;
  } catch (_err) {
    healthPill.textContent = 'Health check unavailable';
  }
}

async function runSearch() {
  const name = document.getElementById('name-input').value.trim();
  const city = document.getElementById('city-input').value.trim();
  const stateValue = document.getElementById('state-input').value.trim();

  if (!name) {
    setSearchMessage('Enter a company name to search.');
    return;
  }

  setSearchMessage('Searching...');

  const params = new URLSearchParams({ name });
  if (city) params.set('city', city);
  if (stateValue) params.set('state', stateValue);
  params.set('limit', '10');

  try {
    const candidates = await fetchJson(`/api/search?${params.toString()}`);
    latestCandidates = candidates || [];
    renderCandidates(latestCandidates);

    if (!latestCandidates.length) {
      setSearchMessage('No matches found. Try a different name or add city/state filters.');
    } else {
      setSearchMessage(`Found ${latestCandidates.length} candidate(s). Choose one to score.`);
    }
  } catch (err) {
    setSearchMessage(`Search failed: ${err.message || 'unknown error'}`);
  }
}

function renderCandidates(candidates) {
  const container = document.getElementById('candidate-list');
  container.innerHTML = '';

  if (!candidates.length) {
    return;
  }

  candidates.forEach((candidate) => {
    const card = document.createElement('article');
    card.className = 'candidate';

    const identity = document.createElement('div');
    identity.className = 'candidate__identity';
    identity.innerHTML = `
      <h3>${escapeHtml(candidate.name || 'Unknown')}</h3>
      <p class="muted">${escapeHtml(formatLocation(candidate.city, candidate.state))}</p>
      <p class="muted">Status: ${escapeHtml(candidate.status || 'Unknown')} · Reviews: ${formatNumber(candidate.review_count)}</p>
      <p class="muted">Last month: ${escapeHtml(candidate.last_review_month || '—')}</p>
    `;

    const meta = document.createElement('div');
    meta.className = 'candidate__meta';

    const availability = document.createElement('p');
    availability.className = 'pill light';
    availability.textContent = candidate.risk_available ? 'Artifact scored' : 'Needs fallback';

    const scoreBtn = document.createElement('button');
    scoreBtn.className = 'solid-btn';
    scoreBtn.type = 'button';
    scoreBtn.textContent = 'Score business';
    scoreBtn.addEventListener('click', async () => {
      await scoreBusiness(candidate.business_id, scoreBtn);
    });

    meta.append(availability, scoreBtn);
    card.append(identity, meta);
    container.append(card);
  });
}

async function scoreBusiness(businessId, buttonEl) {
  if (!businessId) return;
  const forceLiveInference = document.getElementById('force-live-input')?.checked || false;

  const original = buttonEl.textContent;
  buttonEl.textContent = 'Scoring...';
  buttonEl.disabled = true;

  try {
    const result = await fetchJson('/api/score', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        business_id: businessId,
        force_live_inference: forceLiveInference,
      }),
    });

    state.selectedBusinessId = businessId;
    renderScore(result);
  } catch (err) {
    setSearchMessage(`Scoring failed: ${err.message || 'unknown error'}`);
  } finally {
    buttonEl.textContent = original;
    buttonEl.disabled = false;
  }
}

function renderScore(result) {
  const panel = document.getElementById('score-panel');
  panel.classList.remove('hidden');

  document.getElementById('mode-pill').textContent = `Mode: ${result.scoring_mode || 'unknown'}`;
  document.getElementById('biz-name').textContent = result.name || result.business_id || 'Unknown business';
  document.getElementById('biz-meta').textContent = `${formatLocation(result.city, result.state)} · ID: ${result.business_id || '—'}`;
  document.getElementById('biz-status').textContent = `Status: ${result.status || 'Unknown'} · Reviews: ${formatNumber(result.total_reviews)}`;

  document.getElementById('risk-score').textContent =
    typeof result.risk_score === 'number' ? result.risk_score.toFixed(3) : '—';
  document.getElementById('risk-bucket').textContent = result.risk_bucket
    ? `Bucket: ${result.risk_bucket}`
    : 'Bucket: —';

  renderWindows(result.recent_windows || []);
  renderThemes(result.themes_top3 || []);
  renderKeywords(result.problem_keywords);
  renderRecommendations(result.recommendations_top3 || [], result.recommendation_notes);
  renderEvidence(result.evidence_reviews || []);

  const reason = result.not_scored_reason
    ? `Not scored reason: ${result.not_scored_reason}`
    : '';
  document.getElementById('not-scored-reason').textContent = reason;
}

function renderWindows(windows) {
  const list = document.getElementById('windows-list');
  list.innerHTML = '';

  if (!windows.length) {
    list.innerHTML = '<li class="muted">No window-level probabilities available.</li>';
    return;
  }

  windows.forEach((entry) => {
    const item = document.createElement('li');
    const prob = typeof entry.p_closed === 'number' ? entry.p_closed.toFixed(3) : '—';
    item.textContent = `${entry.end_month || '—'} · p_closed=${prob}`;
    list.append(item);
  });
}

function renderThemes(themes) {
  const container = document.getElementById('themes');
  container.innerHTML = '';

  if (!themes.length) {
    container.innerHTML = '<span class="chip muted">No themes</span>';
    return;
  }

  themes.forEach((theme) => {
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.textContent = theme;
    container.append(chip);
  });
}

function renderKeywords(problemKeywords) {
  const el = document.getElementById('keywords');
  el.textContent = problemKeywords || 'No keyword signal available.';
}

function renderRecommendations(recommendations, notes) {
  const list = document.getElementById('recommendations');
  list.innerHTML = '';

  if (!recommendations.length) {
    list.innerHTML = '<li class="muted">No recommendations available.</li>';
  } else {
    recommendations.forEach((rec) => {
      const item = document.createElement('li');
      item.textContent = rec;
      list.append(item);
    });
  }

  document.getElementById('recommendation-notes').textContent = notes || '';
}

function renderEvidence(evidence) {
  const container = document.getElementById('evidence-list');
  container.innerHTML = '';

  if (!evidence.length) {
    container.innerHTML = '<p class="muted">No evidence reviews available for this result.</p>';
    return;
  }

  evidence.forEach((row) => {
    const card = document.createElement('article');
    card.className = 'evidence-item';
    card.innerHTML = `
      <p class="muted">${escapeHtml(row.date || '—')} · stars: ${escapeHtml(String(row.stars ?? '—'))} · neg_prob: ${formatProb(row.sentiment_neg_prob)}</p>
      <p>${escapeHtml(row.snippet || '—')}</p>
    `;
    container.append(card);
  });
}

function setSearchMessage(message) {
  const el = document.getElementById('search-message');
  el.textContent = message;
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);

  let payload = null;
  try {
    payload = await response.json();
  } catch (_err) {
    payload = null;
  }

  if (!response.ok) {
    const message = payload?.message || payload?.error || `Request failed (${response.status})`;
    throw new Error(message);
  }

  return payload;
}

function formatLocation(city, state) {
  const parts = [city, state].filter(Boolean);
  return parts.length ? parts.join(', ') : 'Location unknown';
}

function formatProb(value) {
  return typeof value === 'number' ? value.toFixed(3) : '—';
}

function formatNumber(value) {
  if (typeof value !== 'number') return '—';
  return new Intl.NumberFormat('en-US').format(value);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}
