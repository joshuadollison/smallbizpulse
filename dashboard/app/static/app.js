const numberFmt = new Intl.NumberFormat('en-US');
const percentFmt = new Intl.NumberFormat('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 });
let trendChart;

document.addEventListener('DOMContentLoaded', () => {
  hydrate();
});

async function hydrate() {
  try {
    const [kpis, trend, topics, alerts] = await Promise.all([
      fetchJson('/api/kpis'),
      fetchJson('/api/trend'),
      fetchJson('/api/topics'),
      fetchJson('/api/alerts'),
    ]);

    renderKpis(kpis || []);
    renderTrend(trend || []);
    renderTopics(topics || []);
    renderAlerts(alerts || []);
  } catch (err) {
    console.error('Failed to load placeholder data', err);
    renderError(err.message || 'Unable to load data');
  }
}

async function fetchJson(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Request failed: ${resp.status}`);
  return resp.json();
}

function renderKpis(kpis) {
  const grid = document.getElementById('kpi-grid');
  grid.innerHTML = '';

  if (!kpis.length) {
    grid.innerHTML = '<p class="muted">No KPIs yet. Wire your model outputs to /api/kpis.</p>';
    return;
  }

  kpis.forEach((kpi) => {
    const card = document.createElement('div');
    card.className = 'kpi';

    const label = document.createElement('p');
    label.className = 'label';
    label.textContent = kpi.label;

    const value = document.createElement('p');
    value.className = 'value';
    value.textContent = typeof kpi.value === 'number' ? numberFmt.format(kpi.value) : kpi.value;

    const delta = document.createElement('p');
    delta.className = 'delta';
    const direction = kpi.direction === 'down' ? 'down' : 'up';
    delta.classList.add(direction);
    const sign = kpi.direction === 'down' ? '▼' : '▲';
    delta.textContent = `${sign} ${percentFmt.format(kpi.change || 0)}% vs prior period`;

    card.append(label, value, delta);
    grid.append(card);
  });
}

function renderTrend(trend) {
  const ctx = document.getElementById('trendChart').getContext('2d');

  if (!trend?.length) {
    ctx.font = '14px Work Sans, sans-serif';
    ctx.fillStyle = '#94a3b8';
    ctx.fillText('No trend data yet', 12, 24);
    return;
  }

  const labels = trend.map((row) => row.period);
  const atRisk = trend.map((row) => row.at_risk);
  const closures = trend.map((row) => row.closures);

  if (trendChart) trendChart.destroy();

  trendChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'At-risk',
          data: atRisk,
          borderColor: '#0ea5e9',
          backgroundColor: 'rgba(14, 165, 233, 0.2)',
          tension: 0.35,
          fill: true,
        },
        {
          label: 'Closures',
          data: closures,
          borderColor: '#f59e0b',
          backgroundColor: 'rgba(245, 158, 11, 0.18)',
          tension: 0.35,
          fill: true,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: true, position: 'bottom' } },
      scales: {
        y: { beginAtZero: true, ticks: { precision: 0 } },
        x: { grid: { display: false } },
      },
    },
  });
}

function renderTopics(topics) {
  const container = document.getElementById('topics-list');
  container.innerHTML = '';

  if (!topics.length) {
    container.innerHTML = '<p class="muted">Topic modeling placeholder. Populate /api/topics with your BERTopic output.</p>';
    return;
  }

  topics.forEach((topic) => {
    const card = document.createElement('div');
    card.className = 'topic';

    const title = document.createElement('p');
    title.className = 'topic-title';
    title.textContent = topic.topic;

    const bar = document.createElement('div');
    bar.className = 'bar';
    const fill = document.createElement('span');
    fill.style.width = `${Math.round((topic.weight || 0) * 100)}%`;
    bar.append(fill);

    const rec = document.createElement('p');
    rec.className = 'rec';
    rec.textContent = topic.recommendation;

    const weight = document.createElement('p');
    weight.className = 'muted';
    weight.style.margin = '4px 0 0';
    weight.textContent = `${percentFmt.format((topic.weight || 0) * 100)}% of negative reviews`;

    card.append(title, bar, rec, weight);
    container.append(card);
  });
}

function renderAlerts(alerts) {
  const list = document.getElementById('alerts-list');
  list.innerHTML = '';

  if (!alerts.length) {
    list.innerHTML = '<div class="table__row"><span>No alerts yet.</span></div>';
    return;
  }

  alerts.forEach((alert) => {
    const row = document.createElement('div');
    row.className = 'table__row';

    const riskClass = `risk-${alert.risk}`;
    row.innerHTML = `
      <span>${alert.name}</span>
      <span>${alert.city}</span>
      <span><span class="risk-pill ${riskClass}">${alert.risk}</span></span>
      <span>${alert.sentiment?.toFixed ? alert.sentiment.toFixed(2) : alert.sentiment}</span>
      <span>${alert.lead_issue || '—'}</span>
      <span>${alert.last_review || '—'}</span>
    `;

    list.append(row);
  });
}

function renderError(message) {
  const main = document.querySelector('.main');
  const block = document.createElement('div');
  block.className = 'panel';
  block.innerHTML = `<p class="muted">${message}</p>`;
  main.prepend(block);
}
