import './style.css'
import Plotly from 'plotly.js-dist-min'

//TODO: Have the season selector automatically populate with seasons /episodes from data/seasons
document.querySelector('#app').innerHTML = `
<h1>Survivor ML</h1>
<div class="season-selector"> 
<label for="season-selector">Season:</label>
  <select id="season-selector">
    <option value="20">Season 20</option>
    <option value="50" selected>Season 50</option>
  </select>
</div>
<div class="chart-row">
  <div id="elim-trajectory" class="chart"></div>
  <div id="win-trajectory" class="chart"></div>
</div>
<div class="chart-divider"></div>
<div class="episode-selector">
  <label for="episode-selector">Episode:</label>
  <select id="episode-selector">
    <option value="1">Episode 1</option>
    <option value="2" selected>Episode 2</option>
    <option value="3">Episode 3</option>
    <option value="4">Episode 4</option>
    <option value="5">Episode 5</option>
    <option value="6">Episode 6</option>
    <option value="7">Episode 7</option>
    <option value="8">Episode 8</option>
    <option value="9">Episode 9</option>
    <option value="10">Episode 10</option>
  </select>
</div>
<div class="chart-row">
  <div id="elim-by-episode" class="chart"></div>
  <div id="win-by-episode" class="chart"></div>
</div>

`

async function loadSeason(seasonNumber) {
  const response = await fetch('data/seasons/season_50.json')
  const data = await response.json()



  renderTrajectory(data, 'prob_eliminated', 'Elimination probability by episode', 'P(elimination)', 'elim-trajectory')
  renderTrajectory(data, 'prob_win', 'Win probability by episode', 'P(win)', 'win-trajectory')
  renderBar(data, 1, 'prob_eliminated', 'Elimination probability — Episode 1', 'elim-by-episode')
  renderBar(data, 1, 'prob_win', 'Win probability — Episode 1', 'win-by-episode')
}

function renderTrajectory(data, probCol, title, yLabel, divId) {
  const layout = {
    title: { text: title },
    height: 500,
    xaxis: { title: { text: "Episode" }, dtick: 1 },
    yaxis: { title: { text: yLabel }, tickformat: ".0%" },
    legend: { orientation: 'h', y: -0.2 }
  }

  const traces = data.map(player => ({
    name: player.castaway,
    x: player.episode,
    y: player[probCol],
    mode: 'lines+markers'
  }))

  Plotly.newPlot(divId, traces, layout, { responsive: true })
}

function getEpisodeData(data, episode) {

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

loadSeason('50')

