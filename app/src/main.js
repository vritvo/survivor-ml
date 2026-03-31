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



  renderElimTrajectory(data)
  renderWinTrajectory(data)
}

function renderElimTrajectory(data) {

  const layout = {
    title: "Elimination probability by episode",
    height: 500,
    xaxis: { title: "Episode", dtick: 1 },
    yaxis: { title: "P(elimination)", tickformat: ".0%" },
    legend: { orientation: 'h', y: -0.2 }
  }

  const elimTraces = data.map(player => ({ 
    name: player.castaway, 
    x: player.episode, 
    y: player.prob_eliminated, 
    mode: 'lines+markers'
  }))

  Plotly.newPlot('elim-trajectory', elimTraces, layout, { responsive: true })
}

function renderWinTrajectory(data) {

  const layout = {
    title: "Win probability by episode",
    height: 500,
    xaxis: { title: "Episode", dtick: 1 },
    yaxis: { title: "P(win)", tickformat: ".0%" },
    legend: { orientation: 'h', y: -0.2 }  }

  const winTraces = data.map(player => ({ 
    name: player.castaway, 
    x: player.episode, 
    y: player.prob_win, 
    mode: 'lines+markers'
  }))

  Plotly.newPlot('win-trajectory', winTraces, layout, { responsive: true })
}

loadSeason('50')

