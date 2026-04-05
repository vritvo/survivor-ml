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
      <p id="player-scorecard" class="player-scorecard"></p>
      <h3 class="subsection-title">Feature contributions over time</h3>
      <p class="section-desc">How each feature's contribution to the model score changes episode by episode. Positive values push the prediction higher; negative values push it lower.</p>
      <div class="chart-row">
        <div id="elim-timeline" class="chart"></div>
        <div id="win-timeline" class="chart"></div>
      </div>
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
  bullet: "How this player compares to the field this episode. Dot position = feature value relative to other players. Green = favorable to the player; red = unfavorable.",
  waterfall_elim: "How each feature pushes elimination risk. Green = reduces risk; red = increases it.",
  waterfall_win: "How each feature pushes win probability. Green = increases it; red = decreases it.",
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

  // Default player: winner if season is complete, otherwise most likely winner in final episode
  const winner = data.find(p => p.won_season === 1)
  let defaultPlayer = null
  if (winner) {
    defaultPlayer = winner
  } else {
    const lastEpPlayers = data.filter(p => p.episode.includes(maxEpisode))
    if (lastEpPlayers.length > 0) {
      defaultPlayer = lastEpPlayers.reduce((best, p) => {
        const idx = p.episode.indexOf(maxEpisode)
        const bestIdx = best.episode.indexOf(maxEpisode)
        return p.prob_win[idx] > best.prob_win[bestIdx] ? p : best
      })
    }
  }

  if (defaultPlayer) {
    playerSelect.value = defaultPlayer.castaway_id
  }

  renderTrajectory(data, 'prob_eliminated', 'Elimination probability by episode', 'P(elimination)', 'elim-trajectory')
  renderTrajectory(data, 'prob_win', 'Win probability by episode', 'P(win)', 'win-trajectory')
  renderBar(data, maxEpisode, 'prob_eliminated', 'Elimination probability — Episode ' + maxEpisode, 'elim-by-episode')
  renderBar(data, maxEpisode, 'prob_win', 'Win probability — Episode ' + maxEpisode, 'win-by-episode')

  if (defaultPlayer) {
    renderPlayerDetail(data, defaultPlayer.castaway_id, maxEpisode)
  } else {
    document.getElementById("player-section").style.display = "block"
    document.getElementById("player-empty-state").style.display = "block"
    document.getElementById("player-charts").style.display = "none"
  }
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
    orientation: 'h',
    marker: { color: '#c8873a' }
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

function renderBulletChart(featureData, title, divId, higherIsBetter) {
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

    const contribution = (sortedPlayerVals[i] - sortedAvgs[i]) * sortedCoefs[i]
    const helpsPlayer = higherIsBetter ? contribution > 0 : contribution < 0
    dotX.push(dotPos)
    dotY.push(i)
    dotColors.push(helpsPlayer ? '#3d8b6e' : '#c0604d')
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

  const numFeatures = sortedLabels.length
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
    annotations: [
      { x: -0.9, y: numFeatures - 0.5, yref: 'y', xref: 'x', text: '← Below avg', showarrow: false, font: { size: 11, color: '#999' }, yanchor: 'top' },
      { x: 0.9, y: numFeatures - 0.5, yref: 'y', xref: 'x', text: 'Above avg →', showarrow: false, font: { size: 11, color: '#999' }, yanchor: 'top' },
    ],
    margin: { l: 200, r: 20, t: 40, b: 30 },
  }

  Plotly.newPlot(divId, traces, layout, { responsive: true })
}

function renderWaterfallChart(featureData, title, xLabel, divId, higherIsBetter) {
  const { features, contributions } = featureData

  const indices = features.map((_, i) => i)
  indices.sort((a, b) => contributions[b] - contributions[a])

  const sortedLabels = indices.map(i => FEATURE_LABELS[features[i]] || features[i])
  const sortedContributions = indices.map(i => contributions[i])
  const colors = sortedContributions.map(c => {
    const helpsPlayer = higherIsBetter ? c > 0 : c < 0
    return helpsPlayer ? '#3d8b6e' : '#c0604d'
  })

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

function buildScorecard(data, player) {
  // Compute win rank per episode for this player
  const ranks = []
  const elimRanks = []
  player.episode.forEach((ep, i) => {
    const episodePlayers = data.filter(p => p.episode.includes(ep))
    // Win rank: higher prob = better = lower rank number
    const winProbs = episodePlayers.map(p => p.prob_win[p.episode.indexOf(ep)])
    const playerWinProb = player.prob_win[i]
    const winRank = winProbs.filter(p => p > playerWinProb).length + 1
    ranks.push(winRank)
    // Elim rank: higher prob = worse = higher risk rank
    const elimProbs = episodePlayers.map(p => p.prob_eliminated[p.episode.indexOf(ep)])
    const playerElimProb = player.prob_eliminated[i]
    const elimRank = elimProbs.filter(p => p > playerElimProb).length + 1
    elimRanks.push(elimRank)
  })

  const bestWinRank = Math.min(...ranks)
  const bestWinEp = player.episode[ranks.indexOf(bestWinRank)]
  const timesTop1 = ranks.filter(r => r === 1).length
  const timesTop3 = ranks.filter(r => r <= 3).length
  const totalEps = ranks.length
  const currentWinRank = ranks[ranks.length - 1]
  const currentElimRank = elimRanks[elimRanks.length - 1]
  const lastEp = player.episode[player.episode.length - 1]
  const nPlayersLastEp = data.filter(p => p.episode.includes(lastEp)).length

  const parts = []
  parts.push('<strong>' + player.castaway + '</strong>')

  if (player.won_season === 1) {
    parts.push(' — Winner.')
  } else {
    const wasEliminated = player.eliminated_this_episode.includes(1)
    if (wasEliminated) {
      const elimEpIdx = player.eliminated_this_episode.indexOf(1)
      parts.push(' — Eliminated Episode ' + player.episode[elimEpIdx] + '.')
    } else {
      parts.push(' — Still in the game.')
    }
  }

  parts.push(' Best win rank: #' + bestWinRank + ' (Episode ' + bestWinEp + ').')
  if (timesTop1 > 0) {
    parts.push(' Ranked #1 for ' + timesTop1 + ' of ' + totalEps + ' episodes.')
  }
  if (timesTop3 > 0) {
    parts.push(' Top 3 in ' + timesTop3 + ' of ' + totalEps + ' episodes.')
  }
  parts.push(' Latest win rank: #' + currentWinRank + ' of ' + nPlayersLastEp + '.')

  return parts.join('')
}

function renderContributionTimeline(data, player, modelInfo, featuresKey, title, divId, higherIsBetter) {
  if (!modelInfo) return

  // Exclude final_n
  const excludeFeatures = new Set(["final_n"])
  const featureIndices = modelInfo.features
    .map((f, i) => i)
    .filter(i => !excludeFeatures.has(modelInfo.features[i]))
  const features = featureIndices.map(i => modelInfo.features[i])
  const coefficients = featureIndices.map(i => modelInfo.coefficients[i])
  const scalerMeans = featureIndices.map(i => modelInfo.scaler_means[i])
  const scalerStds = featureIndices.map(i => modelInfo.scaler_stds[i])

  const playerFeatures = player[featuresKey] || player.features

  // Distinct colors that are easy to tell apart
  const palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

  const traces = features.map((f, i) => {
    const contributions = player.episode.map((ep, epIdx) => {
      const val = playerFeatures[f][epIdx]
      const standardized = (val - scalerMeans[i]) / scalerStds[i]
      return coefficients[i] * standardized
    })
    return {
      name: FEATURE_LABELS[f] || f,
      x: player.episode,
      y: contributions,
      mode: 'lines+markers',
      marker: { size: 5 },
      line: { color: palette[i % palette.length] },
    }
  })

  const helpLabel = higherIsBetter ? 'Helps ↑' : 'Hurts ↑'
  const hurtLabel = higherIsBetter ? 'Hurts ↓' : 'Helps ↓'

  const layout = {
    title: { text: title },
    height: 400,
    xaxis: { title: { text: 'Episode' }, dtick: 1 },
    yaxis: { title: { text: 'Contribution' }, zeroline: true, zerolinewidth: 2, zerolinecolor: '#999' },
    legend: { orientation: 'h', y: -0.25, font: { size: 10 } },
    annotations: [
      { x: 1.02, y: 1, xref: 'paper', yref: 'paper', text: helpLabel, showarrow: false, font: { size: 11, color: '#999' }, xanchor: 'left' },
      { x: 1.02, y: 0, xref: 'paper', yref: 'paper', text: hurtLabel, showarrow: false, font: { size: 11, color: '#999' }, xanchor: 'left' },
    ],
    margin: { b: 120, l: 60, r: 70, t: 40 },
  }

  Plotly.newPlot(divId, traces, layout, { responsive: true })
}

function renderPlayerDetail(data, castawayId, episode) {
  const player = data.find(p => p.castaway_id === castawayId)
  if (!player) return

  document.getElementById("player-empty-state").style.display = "none"
  document.getElementById("player-charts").style.display = "block"

  // Render scorecard and timelines (always shown, not episode-dependent)
  document.getElementById("player-scorecard").innerHTML = buildScorecard(data, player)
  renderContributionTimeline(data, player, currentElimModelInfo, 'elim_features', 'Elimination contributions — ' + player.castaway, 'elim-timeline', false)
  renderContributionTimeline(data, player, currentWinModelInfo, 'win_features', 'Win contributions — ' + player.castaway, 'win-timeline', true)

  const epIdx = player.episode.indexOf(episode)
  if (epIdx === -1) {
    const lastEp = player.episode[player.episode.length - 1]
    document.getElementById("view-description").textContent =
      player.castaway + ' was eliminated in Episode ' + lastEp +
      ', but Episode ' + episode + ' is selected. Choose Episode ' + lastEp + ' or earlier to see their breakdown.'
    Plotly.purge('elim-detail')
    Plotly.purge('win-detail')
    return
  }

  const descEl = document.getElementById("view-description")
  if (currentView === 'bullet') {
    descEl.textContent = VIEW_DESCRIPTIONS.bullet
  } else {
    descEl.textContent = "Green = helps the player's chances; red = hurts them. Left chart: elimination risk. Right chart: win probability."
  }

  const elimData = currentElimModelInfo
    ? getPlayerFeatureData(data, player, currentElimModelInfo, 'elim_features', episode)
    : null
  const winData = currentWinModelInfo
    ? getPlayerFeatureData(data, player, currentWinModelInfo, 'win_features', episode)
    : null

  const playerName = player.castaway + " — Episode " + episode

  if (currentView === 'bullet') {
    if (elimData) renderBulletChart(elimData, 'Elimination — ' + playerName, 'elim-detail', false)
    if (winData) renderBulletChart(winData, 'Win — ' + playerName, 'win-detail', true)
  } else {
    if (elimData) renderWaterfallChart(elimData, 'Elimination — ' + playerName, 'Model contribution', 'elim-detail', false)
    if (winData) renderWaterfallChart(winData, 'Win — ' + playerName, 'Model contribution', 'win-detail', true)
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
