import './style.css'
import Plotly from 'plotly.js-dist-min'

document.querySelector('#app').innerHTML = `
<nav class="sidebar">
  <h1>Survivor ML</h1>
  <div class="sidebar-control">
    <label for="season-selector">Season</label>
    <select id="season-selector"></select>
  </div>
  <div class="sidebar-control">
    <label for="episode-selector">Episode</label>
    <select id="episode-selector"></select>
  </div>
</nav>
<main class="content">
  <div class="chart-row">
    <div id="elim-trajectory" class="chart"></div>
    <div id="win-trajectory" class="chart"></div>
  </div>
  <div class="chart-divider"></div>
  <div class="chart-row">
    <div id="elim-by-episode" class="chart"></div>
    <div id="win-by-episode" class="chart"></div>
  </div>
</main>

`

async function loadSeason(seasonNumber) {
  /** Load the season data */

  const response = await fetch('data/seasons/season_' + seasonNumber + '.json')
  const data = await response.json()
  currentData = data

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
  episodeSelect.value = 1

  renderTrajectory(data, 'prob_eliminated', 'Elimination probability by episode', 'P(elimination)', 'elim-trajectory')
  renderTrajectory(data, 'prob_win', 'Win probability by episode', 'P(win)', 'win-trajectory')
  renderBar(data, 1, 'prob_eliminated', 'Elimination probability — Episode 1', 'elim-by-episode')
  renderBar(data, 1, 'prob_win', 'Win probability — Episode 1', 'win-by-episode')
}

function renderTrajectory(data, probCol, title, yLabel, divId) {
  /** Render the trajectory chart for a specific probability column */

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
    // Main line trace
    traces.push({
      name: player.castaway,
      legendgroup: player.castaway,
      x: player.episode,
      y: player[probCol],
      mode: 'lines+markers',
      line: { color: player.won_season === 1 ? 'gold' : undefined, width: player.won_season === 1 ? 3 : 2 },
      marker: { size: player.won_season === 1 ? 8 : 5 }
    })

    // Per-player elimination X marker
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

    // Winner star on final episode
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
  /** Get the data for a specific episode */

  const epData = data.map(player => {
    const idx = player.episode.indexOf(episode)
    if (idx === -1) return null // player wasn't in episode
    return {
      castaway: player.castaway,
      prob_win: player.prob_win[idx],
      prob_eliminated: player.prob_eliminated[idx]    }
    
  }).filter(d => d !== null)

  return epData
}


function renderBar(data, episode, probCol, title, divId) {
  /** Render the bar chart for a specific episode */

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

let currentData = null
var season = '50'
var seasonButton = document.getElementById("season-selector")
seasonButton.addEventListener("change", function(e) {
  season = seasonButton.value
  console.log(season)
  loadSeason(season)
})

var episode = '1'
var episodeButton = document.getElementById("episode-selector")
episodeButton.addEventListener("change", function(e) {
  episode = Number(episodeButton.value)
  console.log(episode)
  renderBar(currentData, episode, 'prob_eliminated', 'Elimination probability — Episode ' + episode, 'elim-by-episode')
  renderBar(currentData, episode, 'prob_win', 'Win probability — Episode ' + episode, 'win-by-episode')

})

// Initialize the app
async function init() {
  const response = await fetch('data/seasons/index.json')
  const seasons = await response.json()

  // Add seasons to the season selector
  const seasonSelect = document.getElementById("season-selector")
  seasons.forEach(season => {
    const option = document.createElement("option")
    option.value = season
    option.textContent = "Season " + season
    seasonSelect.appendChild(option)

  })

  // Default to the highest season
  seasonSelect.value = seasons[seasons.length - 1]

  // Load that season
  loadSeason(seasonSelect.value)
}


init()