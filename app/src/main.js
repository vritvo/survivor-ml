import './style.css'
import Plotly from 'plotly.js-dist-min'

document.querySelector('#app').innerHTML = `
<nav class="sidebar">
  <h1>Survivor ML</h1>
  <p class="sidebar-tagline">Model predictions by season</p>
  <div class="sidebar-control">
    <label for="season-selector">Season</label>
    <select id="season-selector"></select>
  </div>
  <div class="sidebar-control">
    <label for="episode-selector">Episode</label>
    <select id="episode-selector"></select>
  </div>
  <div class="sidebar-control">
    <label for="player-selector">Player</label>
    <select id="player-selector">
      <option value="">Select a player...</option>
    </select>
  </div>
</nav>
<main class="content">
  <section class="dashboard-section">
    <h2 class="section-title">Season Trajectories</h2>
    <p class="section-desc">How each player's predicted probabilities change over the season.</p>
    <div class="chart-row">
      <div id="elim-trajectory" class="chart"></div>
      <div id="win-trajectory" class="chart"></div>
    </div>
  </section>
  <section class="dashboard-section">
    <h2 class="section-title">Episode Snapshot</h2>
    <p class="section-desc">Comparing all players in a single episode.</p>
    <div class="chart-row">
      <div id="elim-by-episode" class="chart"></div>
      <div id="win-by-episode" class="chart"></div>
    </div>
  </section>
  <section id="player-section" class="dashboard-section">
    <h2 class="section-title">Player Breakdown</h2>
    <p class="section-desc">How the model evaluates a specific player.</p>
    <p id="player-empty-state" class="empty-state">Pick a player in the sidebar to explore how the model scores them.</p>
    <div id="player-charts" style="display:none;">
      <div class="detail-header">
        <div class="detail-controls">
          <label>View</label>
          <div class="tab-toggle">
            <button class="tab-btn active" data-view="bullet">Compared to field</button>
            <button class="tab-btn" data-view="waterfall">Model contributions</button>
          </div>
        </div>
      </div>
      <p id="view-description" class="view-description"></p>
      <div class="chart-row">
        <div id="elim-detail" class="chart"></div>
        <div id="win-detail" class="chart"></div>
      </div>
    </div>
  </section>
</main>

`

// Feature display names for readability
const FEATURE_LABELS = {
  "age": "Age",
  "num_previous_seasons": "Previous seasons",
  "votes_against_last_3_eps": "Votes against (last 3 eps)",
  "times_in_danger": "Times in danger",
  "final_n": "Players remaining",
  "confessional_share_rolling_3": "Confessional share (rolling 3)",
  "elim_risk_rank": "Elimination risk rank",
  "mbti_feeling": "Personality: Feeling",
  "advantages_held": "Advantages held",
  "individual_immunity_wins": "Individual immunity wins",
}

const VIEW_DESCRIPTIONS = {
  bullet: "How this player compares to others still in the game this episode. Bar width reflects feature importance; green = favorable for the player, red = unfavorable.",
  waterfall: "How much each feature pushes the model's prediction up or down. Green bars increase the probability, red bars decrease it.",
}

let currentData = null
let currentWinModelInfo = null
let currentElimModelInfo = null
let currentView = 'bullet'

async function loadSeason(seasonNumber) {
  const response = await fetch('data/seasons/season_' + seasonNumber + '.json')
  const json = await response.json()
  const data = json.players || json
  currentData = data
  currentWinModelInfo = json.win_model_info || json.model_info || null
  currentElimModelInfo = json.elim_model_info || null

  // Build episode dropdown from data
  const maxEpisode = Math.max(...data.map(player => player.episode.length))
  const episodeSelect = document.getElementById("episode-selector")
  episodeSelect.innerHTML = ""
  for (let i = 1; i <= maxEpisode; i++) {
    const option = document.createElement("option")
    option.value = i
    option.textContent = "Episode " + i
    episodeSelect.appendChild(option)
  }
  episodeSelect.value = maxEpisode

  // Build player dropdown
  const playerSelect = document.getElementById("player-selector")
  playerSelect.innerHTML = '<option value="">Select a player...</option>'
  const sortedPlayers = [...data].sort((a, b) => a.castaway.localeCompare(b.castaway))
  sortedPlayers.forEach(player => {
    const option = document.createElement("option")
    option.value = player.castaway_id
    option.textContent = player.castaway
    playerSelect.appendChild(option)
  })

  // Show empty state for player section
  document.getElementById("player-section").style.display = "block"
  document.getElementById("player-empty-state").style.display = "block"
  document.getElementById("player-charts").style.display = "none"

  renderTrajectory(data, 'prob_eliminated', 'Elimination probability by episode', 'P(elimination)', 'elim-trajectory')
  renderTrajectory(data, 'prob_win', 'Win probability by episode', 'P(win)', 'win-trajectory')
  renderBar(data, maxEpisode, 'prob_eliminated', 'Elimination probability — Episode ' + maxEpisode, 'elim-by-episode')
  renderBar(data, maxEpisode, 'prob_win', 'Win probability — Episode ' + maxEpisode, 'win-by-episode')
}

function renderTrajectory(data, probCol, title, yLabel, divId) {
  const allY = data.flatMap(player => player[probCol])
  const yMin = Math.min(...allY)
  const yMax = Math.max(...allY)
  const yPad = (yMax - yMin) * 0.05
  const maxEp = Math.max(...data.flatMap(player => player.episode))

  const layout = {
    title: { text: title },
    height: 600,
    xaxis: { title: { text: "Episode" }, dtick: 1, range: [0.5, maxEp + 0.5], autorange: false },
    yaxis: { title: { text: yLabel }, tickformat: ".0%", range: [yMin - yPad, yMax + yPad], autorange: false },
    legend: { orientation: 'h', y: -0.15, font: { size: 10 } },
    margin: { b: 150 }
  }

  const traces = []
  data.forEach(player => {
    traces.push({
      name: player.castaway,
      legendgroup: player.castaway,
      x: player.episode,
      y: player[probCol],
      mode: 'lines+markers',
      line: { color: player.won_season === 1 ? 'gold' : undefined, width: player.won_season === 1 ? 3 : 2 },
      marker: { size: player.won_season === 1 ? 8 : 5 }
    })

    player.eliminated_this_episode.forEach((elim, i) => {
      if (elim === 1) {
        traces.push({
          x: [player.episode[i]],
          y: [player[probCol][i]],
          mode: 'markers',
          marker: { symbol: 'x', size: 8, color: 'rgba(0,0,0,0.5)', line: { width: 1.5 } },
          legendgroup: player.castaway,
          showlegend: false,
          hovertemplate: player.castaway + '<extra>Eliminated</extra>'
        })
      }
    })

    if (player.won_season === 1) {
      const lastIdx = player.episode.length - 1
      traces.push({
        x: [player.episode[lastIdx]],
        y: [player[probCol][lastIdx]],
        mode: 'markers',
        marker: { symbol: 'star', size: 16, color: 'gold', line: { color: 'black', width: 1 } },
        legendgroup: player.castaway,
        showlegend: false,
        hovertemplate: player.castaway + ' - Winner<extra></extra>'
      })
    }
  })

  Plotly.newPlot(divId, traces, layout, { responsive: true })
}

function getEpisodeData(data, episode) {
  const epData = data.map(player => {
    const idx = player.episode.indexOf(episode)
    if (idx === -1) return null
    return {
      castaway: player.castaway,
      prob_win: player.prob_win[idx],
      prob_eliminated: player.prob_eliminated[idx]
    }
  }).filter(d => d !== null)
  return epData
}

function renderBar(data, episode, probCol, title, divId) {
  const layout = {
    title: { text: title },
    height: 500,
    xaxis: { title: { text: "Probability" }, tickformat: ".0%" },
    yaxis: { title: { text: "Player" } },
    margin: { l: 100, t: 40 }
  }

  const epData = getEpisodeData(data, episode)
  const sorted = epData.sort((a, b) => a[probCol] - b[probCol])

  const trace = [{
    x: sorted.map(d => d[probCol]),
    y: sorted.map(d => d.castaway),
    type: 'bar',
    orientation: 'h'
  }]

  Plotly.newPlot(divId, trace, layout, { responsive: true })
}

// --- Player detail rendering ---

function getPlayerFeatureData(data, player, modelInfo, featuresKey, episode) {
  // featuresKey is 'win_features' or 'elim_features' (or 'features' for old format)
  const epIdx = player.episode.indexOf(episode)
  if (epIdx === -1) return null

  // Exclude final_n — identical for all players in an episode
  const excludeFeatures = new Set(["final_n"])
  const featureIndices = modelInfo.features
    .map((f, i) => i)
    .filter(i => !excludeFeatures.has(modelInfo.features[i]))
  const features = featureIndices.map(i => modelInfo.features[i])
  const coefficients = featureIndices.map(i => modelInfo.coefficients[i])
  const scalerMeans = featureIndices.map(i => modelInfo.scaler_means[i])
  const scalerStds = featureIndices.map(i => modelInfo.scaler_stds[i])

  const playerFeatures = player[featuresKey] || player.features
  const playerValues = features.map(f => playerFeatures[f][epIdx])

  // Episode averages, min, max across all remaining players
  const episodePlayers = data.filter(p => p.episode.includes(episode))
  const episodeAvgs = features.map(f => {
    const vals = episodePlayers.map(p => {
      const pFeats = p[featuresKey] || p.features
      return pFeats[f][p.episode.indexOf(episode)]
    })
    return vals.reduce((a, b) => a + b, 0) / vals.length
  })
  const episodeMins = features.map(f => {
    const vals = episodePlayers.map(p => {
      const pFeats = p[featuresKey] || p.features
      return pFeats[f][p.episode.indexOf(episode)]
    })
    return Math.min(...vals)
  })
  const episodeMaxs = features.map(f => {
    const vals = episodePlayers.map(p => {
      const pFeats = p[featuresKey] || p.features
      return pFeats[f][p.episode.indexOf(episode)]
    })
    return Math.max(...vals)
  })

  const contributions = features.map((f, i) => {
    const standardized = (playerValues[i] - scalerMeans[i]) / scalerStds[i]
    return coefficients[i] * standardized
  })

  return { features, coefficients, scalerMeans, scalerStds, playerValues, episodeAvgs, episodeMins, episodeMaxs, contributions }
}

function renderBulletChart(featureData, title, divId) {
  const { features, playerValues, episodeAvgs, episodeMins, episodeMaxs, coefficients } = featureData

  const absCoefs = coefficients.map(c => Math.abs(c))
  const maxAbsCoef = Math.max(...absCoefs)

  const indices = features.map((_, i) => i)
  indices.sort((a, b) => absCoefs[b] - absCoefs[a])

  const sortedLabels = indices.map(i => FEATURE_LABELS[features[i]] || features[i])
  const sortedPlayerVals = indices.map(i => playerValues[i])
  const sortedAvgs = indices.map(i => episodeAvgs[i])
  const sortedMins = indices.map(i => episodeMins[i])
  const sortedMaxs = indices.map(i => episodeMaxs[i])
  const sortedCoefs = indices.map(i => coefficients[i])
  const sortedAbsCoefs = indices.map(i => absCoefs[i])

  const shapes = []
  const dotX = []
  const dotY = []
  const dotColors = []
  const dotHover = []

  sortedLabels.forEach((label, i) => {
    const halfWidth = 0.8 * (sortedAbsCoefs[i] / maxAbsCoef)
    const range = sortedMaxs[i] - sortedMins[i]

    shapes.push({
      type: 'rect',
      x0: -halfWidth, x1: halfWidth,
      y0: i - 0.3, y1: i + 0.3,
      fillcolor: '#e8e8e8',
      line: { color: '#ccc', width: 1 },
      layer: 'below',
    })

    shapes.push({
      type: 'line',
      x0: 0, x1: 0,
      y0: i - 0.3, y1: i + 0.3,
      line: { color: '#999', width: 2 },
      layer: 'below',
    })

    let dotPos = 0
    if (range > 0) {
      const maxDeviation = Math.max(
        sortedMaxs[i] - sortedAvgs[i],
        sortedAvgs[i] - sortedMins[i]
      )
      if (maxDeviation > 0) {
        dotPos = ((sortedPlayerVals[i] - sortedAvgs[i]) / maxDeviation) * halfWidth
      }
    }

    const isGood = (sortedPlayerVals[i] - sortedAvgs[i]) * sortedCoefs[i] > 0
    dotX.push(dotPos)
    dotY.push(i)
    dotColors.push(isGood ? '#2ecc71' : '#e74c3c')
    dotHover.push(label + ': ' + sortedPlayerVals[i].toFixed(2) + ' (avg: ' + sortedAvgs[i].toFixed(2) + ')')
  })

  const traces = [{
    x: dotX,
    y: dotY,
    mode: 'markers',
    marker: { size: 14, color: dotColors, line: { width: 2, color: 'white' } },
    text: dotHover,
    hovertemplate: '%{text}<extra></extra>',
    showlegend: false,
  }]

  const layout = {
    title: { text: title },
    height: 350,
    xaxis: { visible: false, range: [-1.1, 1.1], zeroline: false, showgrid: false },
    yaxis: {
      tickvals: sortedLabels.map((_, i) => i),
      ticktext: sortedLabels,
      autorange: 'reversed',
      automargin: true,
      showgrid: false,
      zeroline: false,
    },
    shapes: shapes,
    margin: { l: 200, r: 20, t: 40, b: 20 },
  }

  Plotly.newPlot(divId, traces, layout, { responsive: true })
}

function renderWaterfallChart(featureData, title, xLabel, divId) {
  const { features, contributions } = featureData

  const indices = features.map((_, i) => i)
  indices.sort((a, b) => contributions[b] - contributions[a])

  const sortedLabels = indices.map(i => FEATURE_LABELS[features[i]] || features[i])
  const sortedContributions = indices.map(i => contributions[i])
  const colors = sortedContributions.map(c => c > 0 ? '#2ecc71' : '#e74c3c')

  const trace = [{
    x: sortedContributions,
    y: sortedLabels,
    type: 'bar',
    orientation: 'h',
    marker: { color: colors },
    hovertemplate: '%{y}: %{x:.3f}<extra></extra>',
  }]

  const layout = {
    title: { text: title },
    height: 350,
    xaxis: { title: { text: xLabel }, zeroline: true, zerolinewidth: 2 },
    yaxis: { automargin: true },
    margin: { l: 200, r: 20, t: 40, b: 40 },
  }

  Plotly.newPlot(divId, trace, layout, { responsive: true })
}

function renderPlayerDetail(data, castawayId, episode) {
  const player = data.find(p => p.castaway_id === castawayId)
  if (!player) return

  const epIdx = player.episode.indexOf(episode)
  if (epIdx === -1) return

  document.getElementById("player-empty-state").style.display = "none"
  document.getElementById("player-charts").style.display = "block"

  // Update description text
  document.getElementById("view-description").textContent = VIEW_DESCRIPTIONS[currentView]

  const elimData = currentElimModelInfo
    ? getPlayerFeatureData(data, player, currentElimModelInfo, 'elim_features', episode)
    : null
  const winData = currentWinModelInfo
    ? getPlayerFeatureData(data, player, currentWinModelInfo, 'win_features', episode)
    : null

  const playerName = player.castaway + " — Episode " + episode

  if (currentView === 'bullet') {
    if (elimData) renderBulletChart(elimData, 'Elimination — ' + playerName, 'elim-detail')
    if (winData) renderBulletChart(winData, 'Win — ' + playerName, 'win-detail')
  } else {
    if (elimData) renderWaterfallChart(elimData, 'Elimination — ' + playerName, 'Contribution (+ favors elimination)', 'elim-detail')
    if (winData) renderWaterfallChart(winData, 'Win — ' + playerName, 'Contribution (+ favors winning)', 'win-detail')
  }
}

// --- Event listeners ---

var seasonButton = document.getElementById("season-selector")
seasonButton.addEventListener("change", function() {
  loadSeason(seasonButton.value)
})

var episodeButton = document.getElementById("episode-selector")
episodeButton.addEventListener("change", function() {
  const episode = Number(episodeButton.value)
  renderBar(currentData, episode, 'prob_eliminated', 'Elimination probability — Episode ' + episode, 'elim-by-episode')
  renderBar(currentData, episode, 'prob_win', 'Win probability — Episode ' + episode, 'win-by-episode')

  const playerId = document.getElementById("player-selector").value
  if (playerId && currentData) {
    renderPlayerDetail(currentData, playerId, episode)
  }
})

var playerButton = document.getElementById("player-selector")
playerButton.addEventListener("change", function() {
  const playerId = playerButton.value
  if (!playerId) {
    document.getElementById("player-empty-state").style.display = "block"
    document.getElementById("player-charts").style.display = "none"
    return
  }
  const episode = Number(document.getElementById("episode-selector").value)
  renderPlayerDetail(currentData, playerId, episode)
})

// Tab toggle
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', function() {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'))
    this.classList.add('active')
    currentView = this.dataset.view

    const playerId = document.getElementById("player-selector").value
    if (playerId && currentData) {
      const episode = Number(document.getElementById("episode-selector").value)
      renderPlayerDetail(currentData, playerId, episode)
    }
  })
})

// Initialize the app
async function init() {
  const response = await fetch('data/seasons/index.json')
  const seasons = await response.json()

  const seasonSelect = document.getElementById("season-selector")
  seasons.forEach(season => {
    const option = document.createElement("option")
    option.value = season
    option.textContent = "Season " + season
    seasonSelect.appendChild(option)
  })

  seasonSelect.value = seasons[seasons.length - 1]
  loadSeason(seasonSelect.value)
}

init()
