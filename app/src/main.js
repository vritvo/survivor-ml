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
    <option value="1" selected>Episode 1</option>
    <option value="2">Episode 2</option>
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
  const response = await fetch('data/seasons/season_' + seasonNumber + '.json')
  const data = await response.json()
  currentData = data

  console.log('data/seasons/season_' + seasonNumber + '.json')
  //todo -- take the sesason and put as argument to functions

  renderTrajectory(data, 'prob_eliminated', 'Elimination probability by episode', 'P(elimination)', 'elim-trajectory')
  renderTrajectory(data, 'prob_win', 'Win probability by episode', 'P(win)', 'win-trajectory')
  renderBar(data, Number(episodeButton.value), 'prob_eliminated', 'Elimination probability — Episode ' + episodeButton.value, 'elim-by-episode')
  renderBar(data, Number(episodeButton.value), 'prob_win', 'Win probability — Episode ' + episodeButton.value, 'win-by-episode')
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

loadSeason(seasonButton.value)