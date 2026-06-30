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
  <div class="sidebar-control model-toggle-control">
    <div class="tab-toggle">
      <button class="tab-btn model-tab active" data-model="win">Win</button>
      <button class="tab-btn model-tab" data-model="elim">Elimination</button>
    </div>
  </div>
</nav>
<main class="content show-win">
  <section class="dashboard-section">
    <h2 class="section-title">Season Trajectories</h2>
    <p class="section-desc">How each player's predicted probabilities change over the season.</p>
    <div class="chart-row">
      <div id="elim-trajectory" class="chart" data-model="elim"></div>
      <div id="win-trajectory" class="chart" data-model="win"></div>
    </div>
    <div class="chart-row chart-row-win-extra">
      <div class="win-contender-aside">
        <p class="contender-aside-label">Why this chart?</p>
        <p class="contender-aside-text">Per-episode P(win) swings a lot. This tracks who the model has ranked #1 most often over the season. It's a smoother read on likely contenders.</p>
      </div>
      <div class="win-extra-column" data-model="win">
        <p id="win-season-pick" class="season-pick-callout"></p>
        <div id="win-cumulative-rank1" class="chart"></div>
      </div>
    </div>
  </section>
  <section class="dashboard-section">
    <h2 class="section-title">Episode Snapshot</h2>
    <p class="section-desc">Comparing all players in a single episode.</p>
    <div class="chart-row">
      <div id="elim-by-episode" class="chart" data-model="elim"></div>
      <div id="win-by-episode" class="chart" data-model="win"></div>
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
        <div id="elim-timeline" class="chart" data-model="elim"></div>
        <div id="win-timeline" class="chart" data-model="win"></div>
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
        <div id="elim-detail" class="chart" data-model="elim"></div>
        <div id="win-detail" class="chart" data-model="win"></div>
      </div>
    </div>
  </section>
  <footer class="site-footer">
    <a href="https://victoriaritvo.com/projects/survivor-ml/" target="_blank" rel="noopener noreferrer">About</a>
    <span class="footer-sep">&middot;</span>
    <a href="https://github.com/vritvo/survivor-ml" target="_blank" rel="noopener noreferrer">GitHub</a>
  </footer>
</main>

`

// Feature display names for readability
const FEATURE_LABELS = {
  "age": "Age",
  "num_previous_seasons": "Previous seasons",
  "vote_accuracy_by_previous_ep": "Vote accuracy (prior eps)",
  "votes_against_last_3_eps": "Votes against (last 3 eps)",
  "times_in_danger": "Times in danger",
  "final_n": "Players remaining",
  "has_advantage": "Has advantage",
  "confessional_share_rolling_3": "Confessional share (rolling 3)",
  "mbti_feeling": "Personality: Feeling",
  "advantages_held": "Advantages held",
  "team_immunity_wins": "Tribal immunity wins",
  "team_immunity_rate": "Tribal immunity win rate",
  "individual_immunity_wins": "Individual immunity wins",
  "individual_immunity_rate": "Individual immunity win rate",
  "immunity_rate": "Immunity win rate (combined)",
  "jury_co_vote_score": "Co-vote alignment with jury",
}

// Model features that map to one display row (e.g. age + age_squared → "Age").
const FEATURE_GROUPS = {
  age: {
    label: "Age",
    members: ["age", "age_squared"],
    displayFeature: "age",
  },
}

const EXCLUDED_DISPLAY_FEATURES = new Set(["final_n"])

function buildDisplayFeatureEntries(modelInfo) {
  const entries = []
  const seenGroups = new Set()

  modelInfo.features.forEach((f, i) => {
    if (EXCLUDED_DISPLAY_FEATURES.has(f)) return

    let groupKey = null
    for (const [key, grp] of Object.entries(FEATURE_GROUPS)) {
      if (grp.members.includes(f)) {
        groupKey = key
        break
      }
    }

    if (groupKey) {
      if (seenGroups.has(groupKey)) return
      seenGroups.add(groupKey)
      const grp = FEATURE_GROUPS[groupKey]
      const memberIndices = modelInfo.features
        .map((feat, idx) => ({ feat, idx }))
        .filter(({ feat }) => grp.members.includes(feat))
        .map(({ idx }) => idx)
      entries.push({
        displayKey: groupKey,
        label: grp.label,
        memberIndices,
        displayFeature: grp.displayFeature,
      })
    } else {
      entries.push({
        displayKey: f,
        label: FEATURE_LABELS[f] || f,
        memberIndices: [i],
        displayFeature: f,
      })
    }
  })

  return entries
}

function standardizedContribution(modelInfo, memberIdx, value) {
  const coef = modelInfo.coefficients[memberIdx]
  const mean = modelInfo.scaler_means[memberIdx]
  const std = modelInfo.scaler_stds[memberIdx]
  return coef * ((value - mean) / std)
}

function combinedContribution(modelInfo, playerFeatures, memberIndices, epIdx) {
  return memberIndices.reduce((sum, idx) => {
    const f = modelInfo.features[idx]
    const series = playerFeatures[f]
    if (!series) return sum
    const val = series[epIdx]
    if (val == null || Number.isNaN(val)) return sum
    return sum + standardizedContribution(modelInfo, idx, val)
  }, 0)
}

/**
 * Comparable per-feature importance, in log-odds units.
 *
 * - Linear feature: |β| — log-odds change per 1 SD of the feature.
 * - Quadratic group (e.g. age + age_squared): rewrite β_lin·z + β_sq·z² as the
 *   curve c(d−x)², where c = β_sq / σ_sq is the raw-space curvature. The bar uses
 *   |c|·σ_lin² = the log-odds swing for a 1-SD move away from the peak. 
 */
function comparableDisplayImportance(modelInfo, memberIndices) {
  if (memberIndices.length === 1) {
    return Math.abs(modelInfo.coefficients[memberIndices[0]])
  }

  let idxLin = null
  let idxSq = null
  for (const idx of memberIndices) {
    if (modelInfo.features[idx].endsWith("_squared")) idxSq = idx
    else idxLin = idx
  }

  if (idxLin != null && idxSq != null) {
    const sdLin = modelInfo.scaler_stds[idxLin]
    const sdSq = modelInfo.scaler_stds[idxSq]
    if (sdSq === 0) return Math.abs(modelInfo.coefficients[idxLin])
    const c = modelInfo.coefficients[idxSq] / sdSq // raw-space curvature
    return Math.abs(c) * sdLin * sdLin
  }

  return memberIndices.reduce(
    (sum, idx) => sum + Math.abs(modelInfo.coefficients[idx]),
    0,
  )
}

function getAgePeakYears(modelInfo) {
  const idxAge = modelInfo.features.indexOf("age")
  const idxSq = modelInfo.features.indexOf("age_squared")
  if (idxAge === -1 || idxSq === -1) return null

  const b1 = modelInfo.coefficients[idxAge]
  const b2 = modelInfo.coefficients[idxSq]
  const s1 = modelInfo.scaler_stds[idxAge]
  const s2 = modelInfo.scaler_stds[idxSq]
  if (b2 === 0) return null

  const peak = -(b1 * s2) / (2 * b2 * s1)
  if (!Number.isFinite(peak) || peak < 18 || peak > 75) return null
  return Math.round(peak)
}

function ageHoverSuffix(modelInfo) {
  const peak = getAgePeakYears(modelInfo)
  return peak != null ? ` · model peak ~${peak} yrs` : ""
}

const VIEW_DESCRIPTIONS = {
  bullet: "Bar width ≈ how much the model weighs each feature overall (comparable coefficients, same for every player). Dot position = this player's value vs the rest of the cast; green/red = whether it helps or hurts them.",
  waterfall_elim: "How each feature pushes elimination risk. Green = reduces risk; red = increases it.",
  waterfall_win: "How each feature pushes win probability. Green = increases it; red = decreases it.",
}

let currentData = null
let currentWinModelInfo = null
let currentElimModelInfo = null
let currentView = 'bullet'

const MOBILE_BREAKPOINT = 768
function isMobile() { return window.innerWidth <= MOBILE_BREAKPOINT }

// Cumulative #1 share is noisy for the first few episodes (one good week = 100%).
const CUMULATIVE_RANK_MIN_EPISODE = 6

const WINNER_COLOR = 'gold'
const ACCENT_COLOR = '#c8873a'
const SELECTED_BAR_COLOR = '#8b5a2b'
const BACKGROUND_OPACITY = 0.72

// Plotly default qualitative palette — assigned by player name, not trace order.
const TRACE_PALETTE = [
  '#636efa', '#EF553B', '#00cc96', '#ab63fa', '#FFA15A',
  '#19d3f3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52',
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]

let currentPlayerColorMap = {}

function buildPlayerColorMap(data) {
  const sorted = [...data].sort((a, b) => a.castaway.localeCompare(b.castaway))
  const map = {}
  sorted.forEach((p, i) => {
    map[p.castaway_id] = TRACE_PALETTE[i % TRACE_PALETTE.length]
  })
  return map
}

/** Background players first (alpha), then winner, then selected — both on top. */
function orderPlayersForDraw(data, selectedId) {
  const winner = data.find(p => p.won_season === 1)
  const winnerId = winner?.castaway_id

  const background = data.filter(p => {
    if (p.won_season === 1) return false
    if (selectedId && p.castaway_id === selectedId) return false
    return true
  })

  const foreground = []
  if (winner && winnerId !== selectedId) foreground.push(winner)
  if (selectedId) {
    const selected = data.find(p => p.castaway_id === selectedId)
    if (selected) foreground.push(selected)
  } else if (winner) {
    foreground.push(winner)
  }

  return [
    ...background.sort((a, b) => a.castaway.localeCompare(b.castaway)),
    ...foreground,
  ]
}

function orderContendersForDraw(contenders, selectedId) {
  const winner = contenders.find(s => s.won_season === 1)
  const winnerId = winner?.castaway_id

  const background = contenders.filter(s => {
    if (s.won_season === 1) return false
    if (selectedId && s.castaway_id === selectedId) return false
    return true
  })

  const foreground = []
  if (winner && winnerId !== selectedId) foreground.push(winner)
  if (selectedId) {
    const selected = contenders.find(s => s.castaway_id === selectedId)
    if (selected) foreground.push(selected)
  } else if (winner) {
    foreground.push(winner)
  }

  return [
    ...background.sort((a, b) => a.castaway.localeCompare(b.castaway)),
    ...foreground,
  ]
}

function getSelectedPlayerId() {
  return document.getElementById('player-selector').value || null
}

function trajectoryTraceStyle(player, selectedId, colorMap) {
  const isWinner = player.won_season === 1
  const isSelected = selectedId && player.castaway_id === selectedId
  const isHighlighted = isWinner || isSelected

  let lineColor = colorMap[player.castaway_id]
  let lineWidth = 2
  let markerSize = 5
  let opacity = isHighlighted ? 1 : BACKGROUND_OPACITY

  if (isWinner) {
    lineColor = WINNER_COLOR
    lineWidth = 4
    markerSize = 9
  }
  if (isSelected && !isWinner) {
    lineColor = ACCENT_COLOR
    lineWidth = 5
    markerSize = 10
  }
  if (isSelected && isWinner) {
    lineWidth = 5
    markerSize = 11
  }

  const line = { width: lineWidth, color: lineColor }
  return { line, markerSize, opacity }
}

function refreshOverviewCharts(episode) {
  if (!currentData) return
  renderTrajectory(currentData, 'prob_eliminated', 'Elimination probability by episode', 'P(elimination)', 'elim-trajectory', episode)
  renderTrajectory(currentData, 'prob_win', 'Win probability by episode', 'P(win)', 'win-trajectory', episode)
  renderCumulativeRank1(currentData, episode)
  renderBar(currentData, episode, 'prob_eliminated', 'Elimination probability — Episode ' + episode, 'elim-by-episode')
  renderBar(currentData, episode, 'prob_win', 'Win probability — Episode ' + episode, 'win-by-episode')
}

async function loadSeason(seasonNumber) {
  const response = await fetch('data/seasons/season_' + seasonNumber + '.json')
  const json = await response.json()
  const data = json.players || json
  currentData = data
  currentPlayerColorMap = buildPlayerColorMap(data)
  currentWinModelInfo = json.win_model_info || json.model_info || null
  currentElimModelInfo = json.elim_model_info || null

  // Build episode dropdown from data (use max episode number, not array length —
  // some seasons skip episode numbers, e.g. season 6 has no episode 10)
  const maxEpisode = Math.max(...data.flatMap(player => player.episode))
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

  refreshOverviewCharts(maxEpisode)

  if (defaultPlayer) {
    renderPlayerDetail(data, defaultPlayer.castaway_id, maxEpisode)
  } else {
    document.getElementById("player-section").style.display = "block"
    document.getElementById("player-empty-state").style.display = "block"
    document.getElementById("player-charts").style.display = "none"
  }
}

function renderTrajectory(data, probCol, title, yLabel, divId, maxEpisode) {
  const seasonMaxEp = Math.max(...data.flatMap(player => player.episode))
  const effectiveMaxEp = maxEpisode ?? seasonMaxEp
  const selectedId = getSelectedPlayerId()

  const allY = data.flatMap(player => player[probCol])
  const yMin = Math.min(...allY)
  const yMax = Math.max(...allY)
  const yPad = (yMax - yMin) * 0.05

  const displayTitle = effectiveMaxEp < seasonMaxEp
    ? title + ' (through Episode ' + effectiveMaxEp + ')'
    : title

  const mobile = isMobile()
  const layout = {
    title: { text: displayTitle, font: { size: mobile ? 13 : 17 } },
    height: mobile ? 350 : 600,
    xaxis: { title: { text: "Episode" }, dtick: 1, range: [0.5, seasonMaxEp + 0.5], autorange: false },
    yaxis: { title: { text: yLabel }, tickformat: ".0%", range: [yMin - yPad, yMax + yPad], autorange: false },
    legend: { orientation: 'h', y: mobile ? -0.3 : -0.15, font: { size: mobile ? 9 : 10 } },
    margin: { b: mobile ? 100 : 150, l: mobile ? 50 : 80, r: mobile ? 10 : 80, t: mobile ? 30 : 40 }
  }

  const traces = []
  orderPlayersForDraw(data, selectedId).forEach(player => {
    const episodes = []
    const probs = []
    for (let i = 0; i < player.episode.length; i++) {
      if (player.episode[i] > effectiveMaxEp) break
      episodes.push(player.episode[i])
      probs.push(player[probCol][i])
    }
    if (episodes.length === 0) return

    const style = trajectoryTraceStyle(player, selectedId, currentPlayerColorMap)
    traces.push({
      name: player.castaway,
      legendgroup: player.castaway,
      x: episodes,
      y: probs,
      mode: 'lines+markers',
      opacity: style.opacity,
      line: style.line,
      marker: { size: style.markerSize, opacity: style.opacity },
    })

    for (let i = 0; i < player.episode.length; i++) {
      if (player.episode[i] > effectiveMaxEp) break
      if (player.eliminated_this_episode[i] === 1) {
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
    }

    if (player.won_season === 1) {
      const finaleEp = player.episode[player.episode.length - 1]
      if (finaleEp <= effectiveMaxEp) {
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
    }
  })

  Plotly.newPlot(divId, traces, layout, { responsive: true })
}

// --- Cumulative #1 rank (win model readout) ---

function rankPlayersAtEpisode(playersAtEp) {
  const sorted = [...playersAtEp].sort((a, b) => {
    if (b.prob_win !== a.prob_win) return b.prob_win - a.prob_win
    return a.castaway_id.localeCompare(b.castaway_id)
  })
  const ranks = new Map()
  sorted.forEach((row, i) => ranks.set(row.castaway_id, i + 1))
  return ranks
}

function computeCumulativeRankStats(data, maxEpisode) {
  const seasonMaxEp = Math.max(...data.flatMap(player => player.episode))
  const effectiveMaxEp = maxEpisode ?? seasonMaxEp

  const episodeNumbers = [...new Set(data.flatMap(p => p.episode))]
    .filter(ep => ep <= effectiveMaxEp)
    .sort((a, b) => a - b)

  const playerStats = {}
  data.forEach(p => {
    playerStats[p.castaway_id] = {
      castaway: p.castaway,
      castaway_id: p.castaway_id,
      won_season: p.won_season,
      episodes: [],
      fracRank1: [],
      cumRank1: 0,
      epCount: 0,
      everRank1: false,
    }
  })

  episodeNumbers.forEach(ep => {
    const atEp = data
      .map(p => {
        const idx = p.episode.indexOf(ep)
        if (idx === -1) return null
        return {
          castaway_id: p.castaway_id,
          castaway: p.castaway,
          prob_win: p.prob_win[idx],
        }
      })
      .filter(row => row !== null)

    const ranks = rankPlayersAtEpisode(atEp)
    atEp.forEach(row => {
      const stats = playerStats[row.castaway_id]
      stats.epCount += 1
      if (ranks.get(row.castaway_id) === 1) {
        stats.cumRank1 += 1
        stats.everRank1 = true
      }
      stats.episodes.push(ep)
      stats.fracRank1.push(stats.cumRank1 / stats.epCount)
    })
  })

  return { playerStats, seasonMaxEp, effectiveMaxEp }
}

function getSeasonLongPick(stats) {
  const { playerStats, effectiveMaxEp } = stats
  let best = null

  Object.values(playerStats).forEach(s => {
    const epIdx = s.episodes.indexOf(effectiveMaxEp)
    if (epIdx === -1) return
    const frac = s.fracRank1[epIdx]
    if (best === null || frac > best.frac ||
        (frac === best.frac && s.cumRank1 > best.cumRank1)) {
      best = {
        castaway: s.castaway,
        frac,
        cumRank1: s.cumRank1,
        epCount: s.epCount,
      }
    }
  })

  return best
}

function renderCumulativeRank1(data, maxEpisode) {
  const stats = computeCumulativeRankStats(data, maxEpisode)
  const { playerStats, seasonMaxEp, effectiveMaxEp } = stats
  const selectedId = getSelectedPlayerId()
  const rowEl = document.querySelector('.chart-row-win-extra')

  if (effectiveMaxEp < CUMULATIVE_RANK_MIN_EPISODE) {
    rowEl.style.display = 'none'
    Plotly.purge('win-cumulative-rank1')
    return
  }
  rowEl.style.display = ''

  const pickEl = document.getElementById('win-season-pick')
  const pick = getSeasonLongPick(stats)
  if (pick && pick.cumRank1 > 0) {
    const throughLabel = effectiveMaxEp < seasonMaxEp
      ? ' (through Episode ' + effectiveMaxEp + ')'
      : ''
    pickEl.innerHTML = '<strong>Model\u2019s season pick' + throughLabel + ':</strong> ' +
      pick.castaway + ' \u2014 ranked #1 in ' + pick.cumRank1 + ' of ' + pick.epCount + ' episodes'
    pickEl.style.display = 'block'
  } else {
    pickEl.style.display = 'none'
  }

  const contenders = Object.values(playerStats).filter(
    s => s.everRank1 || s.won_season === 1 || (selectedId && s.castaway_id === selectedId)
  )

  const displayTitle = effectiveMaxEp < seasonMaxEp
    ? 'Fraction of episodes ranked #1 (through Episode ' + effectiveMaxEp + ')'
    : 'Fraction of episodes ranked #1 (cumulative)'

  const mobile = isMobile()
  const layout = {
    title: { text: displayTitle, font: { size: mobile ? 13 : 17 } },
    height: mobile ? 320 : 450,
    xaxis: { title: { text: 'Episode' }, dtick: 1, range: [0.5, seasonMaxEp + 0.5], autorange: false },
    yaxis: { title: { text: 'Frac. ranked #1' }, tickformat: '.0%', range: [0, 1.05], autorange: false },
    legend: { orientation: 'h', y: mobile ? -0.35 : -0.2, font: { size: mobile ? 9 : 10 } },
    margin: { b: mobile ? 100 : 130, l: mobile ? 50 : 80, r: mobile ? 10 : 80, t: mobile ? 30 : 40 },
  }

  const traces = []
  orderContendersForDraw(contenders, selectedId).forEach(s => {
    if (s.episodes.length === 0) return

    const player = { castaway_id: s.castaway_id, won_season: s.won_season }
    const style = trajectoryTraceStyle(player, selectedId, currentPlayerColorMap)
    traces.push({
      name: s.castaway,
      legendgroup: s.castaway,
      x: s.episodes,
      y: s.fracRank1,
      mode: 'lines+markers',
      opacity: style.opacity,
      line: style.line,
      marker: { size: style.markerSize, opacity: style.opacity },
    })

    if (s.won_season === 1) {
      const finaleEp = s.episodes[s.episodes.length - 1]
      if (finaleEp <= effectiveMaxEp) {
        const lastIdx = s.episodes.length - 1
        traces.push({
          x: [s.episodes[lastIdx]],
          y: [s.fracRank1[lastIdx]],
          mode: 'markers',
          marker: { symbol: 'star', size: 16, color: 'gold', line: { color: 'black', width: 1 } },
          legendgroup: s.castaway,
          showlegend: false,
          hovertemplate: s.castaway + ' - Winner<extra></extra>',
        })
      }
    }
  })

  Plotly.newPlot('win-cumulative-rank1', traces, layout, { responsive: true })
}

function getEpisodeData(data, episode) {
  const epData = data.map(player => {
    const idx = player.episode.indexOf(episode)
    if (idx === -1) return null
    return {
      castaway_id: player.castaway_id,
      castaway: player.castaway,
      won_season: player.won_season,
      prob_win: player.prob_win[idx],
      prob_eliminated: player.prob_eliminated[idx]
    }
  }).filter(d => d !== null)
  return epData
}

function barColorForPlayer(row, selectedId) {
  const isSelected = selectedId && row.castaway_id === selectedId
  const isWinner = row.won_season === 1
  const isHighlighted = isSelected || isWinner
  if (isSelected && isWinner) return WINNER_COLOR
  if (isSelected) return SELECTED_BAR_COLOR
  if (isWinner) return WINNER_COLOR
  if (selectedId && !isHighlighted) return 'rgba(200, 135, 58, ' + BACKGROUND_OPACITY + ')'
  return ACCENT_COLOR
}

function renderBar(data, episode, probCol, title, divId) {
  const selectedId = getSelectedPlayerId()
  const mobile = isMobile()
  const layout = {
    title: { text: title, font: { size: mobile ? 13 : 17 } },
    height: mobile ? 350 : 500,
    xaxis: { title: { text: "Probability" }, tickformat: ".0%" },
    yaxis: { title: { text: "Player" }, tickfont: { size: mobile ? 10 : 12 } },
    margin: { l: mobile ? 80 : 100, t: mobile ? 30 : 40, r: mobile ? 10 : 80 }
  }

  const epData = getEpisodeData(data, episode)
  const sorted = epData.sort((a, b) => a[probCol] - b[probCol])

  const trace = [{
    x: sorted.map(d => d[probCol]),
    y: sorted.map(d => d.castaway),
    type: 'bar',
    orientation: 'h',
    marker: { color: sorted.map(d => barColorForPlayer(d, selectedId)) }
  }]

  Plotly.newPlot(divId, trace, layout, { responsive: true })
}

// --- Player detail rendering ---

function getPlayerFeatureData(data, player, modelInfo, featuresKey, episode) {
  // featuresKey is 'win_features' or 'elim_features' (or 'features' for old format)
  const epIdx = player.episode.indexOf(episode)
  if (epIdx === -1) return null

  const playerFeatures = player[featuresKey] || player.features
  const entries = buildDisplayFeatureEntries(modelInfo)
  const features = entries.map(e => e.displayKey)
  const labels = entries.map(e => e.label)

  const playerValues = entries.map(e => playerFeatures[e.displayFeature][epIdx])

  // Episode averages, min, max across all remaining players (use display feature for groups)
  const episodePlayers = data.filter(p => p.episode.includes(episode))
  const episodeAvgs = entries.map(e => {
    const vals = episodePlayers.map(p => {
      const pFeats = p[featuresKey] || p.features
      return pFeats[e.displayFeature][p.episode.indexOf(episode)]
    })
    return vals.reduce((a, b) => a + b, 0) / vals.length
  })
  const episodeMins = entries.map(e => {
    const vals = episodePlayers.map(p => {
      const pFeats = p[featuresKey] || p.features
      return pFeats[e.displayFeature][p.episode.indexOf(episode)]
    })
    return Math.min(...vals)
  })
  const episodeMaxs = entries.map(e => {
    const vals = episodePlayers.map(p => {
      const pFeats = p[featuresKey] || p.features
      return pFeats[e.displayFeature][p.episode.indexOf(episode)]
    })
    return Math.max(...vals)
  })

  const contributions = entries.map(e =>
    combinedContribution(modelInfo, playerFeatures, e.memberIndices, epIdx)
  )
  const importances = entries.map(e =>
    comparableDisplayImportance(modelInfo, e.memberIndices)
  )

  return {
    features,
    labels,
    modelInfo,
    playerValues,
    episodeAvgs,
    episodeMins,
    episodeMaxs,
    contributions,
    importances,
  }
}

function renderBulletChart(featureData, title, divId, higherIsBetter) {
  const {
    labels, playerValues, episodeAvgs, episodeMins, episodeMaxs,
    contributions, importances, modelInfo, features,
  } = featureData

  const maxImportance = Math.max(...importances, 1e-9)

  const indices = labels.map((_, i) => i)
  indices.sort((a, b) => importances[b] - importances[a])

  const sortedLabels = indices.map(i => labels[i])
  const sortedPlayerVals = indices.map(i => playerValues[i])
  const sortedAvgs = indices.map(i => episodeAvgs[i])
  const sortedMins = indices.map(i => episodeMins[i])
  const sortedMaxs = indices.map(i => episodeMaxs[i])
  const sortedContributions = indices.map(i => contributions[i])
  const sortedImportances = indices.map(i => importances[i])
  const sortedFeatures = indices.map(i => features[i])

  const shapes = []
  const dotX = []
  const dotY = []
  const dotColors = []
  const dotHover = []

  sortedLabels.forEach((label, i) => {
    const halfWidth = 0.8 * (sortedImportances[i] / maxImportance)
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

    const helpsPlayer = higherIsBetter
      ? sortedContributions[i] > 0
      : sortedContributions[i] < 0
    dotX.push(dotPos)
    dotY.push(i)
    dotColors.push(helpsPlayer ? '#3d8b6e' : '#c0604d')
    const ageSuffix = sortedFeatures[i] === "age" ? ageHoverSuffix(modelInfo) : ""
    dotHover.push(
      label + ': ' + sortedPlayerVals[i].toFixed(1)
      + ' (avg: ' + sortedAvgs[i].toFixed(1) + ')'
      + ageSuffix
    )
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

  const mobile = isMobile()
  const numFeatures = sortedLabels.length
  const layout = {
    title: { text: title, font: { size: mobile ? 12 : 17 } },
    height: mobile ? 300 : 350,
    xaxis: { visible: false, range: [-1.1, 1.1], zeroline: false, showgrid: false },
    yaxis: {
      tickvals: sortedLabels.map((_, i) => i),
      ticktext: sortedLabels,
      tickfont: { size: mobile ? 10 : 12 },
      autorange: 'reversed',
      automargin: true,
      showgrid: false,
      zeroline: false,
    },
    shapes: shapes,
    annotations: [
      { x: -0.9, y: numFeatures - 0.5, yref: 'y', xref: 'x', text: '← Below avg', showarrow: false, font: { size: mobile ? 9 : 11, color: '#999' }, yanchor: 'top' },
      { x: 0.9, y: numFeatures - 0.5, yref: 'y', xref: 'x', text: 'Above avg →', showarrow: false, font: { size: mobile ? 9 : 11, color: '#999' }, yanchor: 'top' },
    ],
    margin: { l: mobile ? 120 : 200, r: mobile ? 10 : 20, t: mobile ? 30 : 40, b: 30 },
  }

  Plotly.newPlot(divId, traces, layout, { responsive: true })
}

function renderWaterfallChart(featureData, title, xLabel, divId, higherIsBetter) {
  const { labels, contributions, features, modelInfo, playerValues } = featureData

  const indices = labels.map((_, i) => i)
  indices.sort((a, b) => contributions[b] - contributions[a])

  const sortedLabels = indices.map(i => labels[i])
  const sortedContributions = indices.map(i => contributions[i])
  const sortedHover = indices.map(i => {
    let text = `${labels[i]}: ${contributions[i].toFixed(3)}`
    if (features[i] === "age") {
      text += ` (${playerValues[i]} yrs${ageHoverSuffix(modelInfo)})`
    }
    return text
  })
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
    customdata: sortedHover,
    hovertemplate: '%{customdata}<extra></extra>',
  }]

  const mobile = isMobile()
  const numFeatures = sortedLabels.length
  const layout = {
    title: { text: title, font: { size: mobile ? 12 : 17 } },
    height: mobile ? Math.max(300, 48 * numFeatures) : Math.max(350, 42 * numFeatures),
    xaxis: { title: { text: xLabel }, zeroline: true, zerolinewidth: 2 },
    yaxis: { automargin: true, tickfont: { size: mobile ? 10 : 12 } },
    margin: { l: mobile ? 10 : 10, r: mobile ? 24 : 32, t: mobile ? 30 : 40, b: 40 },
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

  const playerFeatures = player[featuresKey] || player.features
  const entries = buildDisplayFeatureEntries(modelInfo)

  const palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

  const traces = entries.map((entry, i) => {
    const contributions = player.episode.map((_, epIdx) =>
      combinedContribution(modelInfo, playerFeatures, entry.memberIndices, epIdx)
    )
    return {
      name: entry.label,
      x: player.episode,
      y: contributions,
      mode: 'lines+markers',
      marker: { size: 5 },
      line: { color: palette[i % palette.length] },
    }
  })

  const mobile = isMobile()
  const helpLabel = higherIsBetter ? 'Helps ↑' : 'Hurts ↑'
  const hurtLabel = higherIsBetter ? 'Hurts ↓' : 'Helps ↓'

  const layout = {
    title: { text: title, font: { size: mobile ? 12 : 17 } },
    height: mobile ? 300 : 400,
    xaxis: { title: { text: 'Episode' }, dtick: 1 },
    yaxis: { title: { text: 'Contribution' }, zeroline: true, zerolinewidth: 2, zerolinecolor: '#999' },
    legend: { orientation: 'h', y: mobile ? -0.4 : -0.25, font: { size: mobile ? 9 : 10 } },
    annotations: mobile ? [] : [
      { x: 1.02, y: 1, xref: 'paper', yref: 'paper', text: helpLabel, showarrow: false, font: { size: 11, color: '#999' }, xanchor: 'left' },
      { x: 1.02, y: 0, xref: 'paper', yref: 'paper', text: hurtLabel, showarrow: false, font: { size: 11, color: '#999' }, xanchor: 'left' },
    ],
    margin: { b: mobile ? 80 : 120, l: mobile ? 45 : 60, r: mobile ? 10 : 70, t: mobile ? 30 : 40 },
  }

  Plotly.newPlot(divId, traces, layout, { responsive: true })
}

function renderDetailPlaceholder(divId, chartTitle, message) {
  const mobile = isMobile()
  Plotly.newPlot(divId, [], {
    title: { text: chartTitle, font: { size: mobile ? 12 : 17 } },
    height: mobile ? 300 : 350,
    paper_bgcolor: '#faf9f7',
    plot_bgcolor: '#f5f3f0',
    xaxis: { visible: false, fixedrange: true },
    yaxis: { visible: false, fixedrange: true },
    annotations: [{
      text: message,
      xref: 'paper',
      yref: 'paper',
      x: 0.5,
      y: 0.5,
      showarrow: false,
      font: { size: mobile ? 13 : 15, color: '#5c5349' },
      align: 'center',
    }],
    shapes: [{
      type: 'rect',
      xref: 'paper',
      yref: 'paper',
      x0: 0.03,
      y0: 0.03,
      x1: 0.97,
      y1: 0.97,
      line: { color: '#c8873a', width: 1.5, dash: 'dot' },
      fillcolor: 'rgba(255, 255, 255, 0.85)',
      layer: 'below',
    }],
    margin: { l: 28, r: 28, t: mobile ? 36 : 44, b: 28 },
  }, { responsive: true, displayModeBar: false })
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
  const descEl = document.getElementById("view-description")
  if (epIdx === -1) {
    const lastEp = Math.max(...player.episode)
    const placeholderMsg =
      player.castaway + ' was eliminated in Episode ' + lastEp + ',<br>' +
      'but Episode ' + episode + ' is selected.<br><br>' +
      'Choose Episode ' + lastEp + ' or earlier to see their breakdown.'
    descEl.style.display = 'none'
    renderDetailPlaceholder('elim-detail', 'Elimination — ' + player.castaway, placeholderMsg)
    renderDetailPlaceholder('win-detail', 'Win — ' + player.castaway, placeholderMsg)
    return
  }

  descEl.style.display = ''
  if (currentView === 'bullet') {
    descEl.textContent = VIEW_DESCRIPTIONS.bullet
  } else {
    descEl.textContent = "Green = helps the player's chances; red = hurts them."
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
  refreshOverviewCharts(episode)

  const playerId = document.getElementById("player-selector").value
  if (playerId && currentData) {
    renderPlayerDetail(currentData, playerId, episode)
  }
})

var playerButton = document.getElementById("player-selector")
playerButton.addEventListener("change", function() {
  const playerId = playerButton.value
  const episode = Number(document.getElementById("episode-selector").value)

  if (!playerId) {
    document.getElementById("player-empty-state").style.display = "block"
    document.getElementById("player-charts").style.display = "none"
    refreshOverviewCharts(episode)
    return
  }

  refreshOverviewCharts(episode)
  renderPlayerDetail(currentData, playerId, episode)
})

// Tab toggle
document.querySelectorAll('.tab-btn[data-view]').forEach(btn => {
  btn.addEventListener('click', function() {
    document.querySelectorAll('.tab-btn[data-view]').forEach(b => b.classList.remove('active'))
    this.classList.add('active')
    currentView = this.dataset.view

    const playerId = document.getElementById("player-selector").value
    if (playerId && currentData) {
      const episode = Number(document.getElementById("episode-selector").value)
      renderPlayerDetail(currentData, playerId, episode)
    }
  })
})

// Model toggle (mobile only, but wired up always)
document.querySelectorAll('.model-tab').forEach(btn => {
  btn.addEventListener('click', function() {
    document.querySelectorAll('.model-tab').forEach(b => b.classList.remove('active'))
    this.classList.add('active')
    const model = this.dataset.model
    const content = document.querySelector('.content')
    content.classList.remove('show-win', 'show-elim')
    content.classList.add('show-' + model)
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
