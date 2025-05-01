// Global variables
let returnChart = null;
let weightChart = null;
let concentrationChart = null;
let sectorData = null;
let selectedTickers = new Set();
let colorMap = {};  // Store color assignments for tickers

// Register the annotation plugin if it's available
if (typeof Chart !== 'undefined' && Chart.annotation) {
  Chart.register(Chart.annotation);
}

// Fetch tickers grouped by sector
async function fetchTickersBySector() {
  try {
    const response = await fetch('/api/tickers_by_sector');
    return await response.json();
  } catch (error) {
    console.error('Error fetching tickers:', error);
    return [];
  }
}

// Format date to YYYY-MM-DD
function formatDate(date) {
  return date.toISOString().split('T')[0];
}

// Set active navigation link
function setActiveNavLink() {
  const currentPath = window.location.pathname;
  const navLinks = document.querySelectorAll('.nav-link');
  
  navLinks.forEach(link => {
    const linkPath = link.getAttribute('href');
    if (currentPath.endsWith(linkPath) || 
        (currentPath.endsWith('/') && linkPath === '/index.html') ||
        (currentPath.endsWith('/fullpage') && linkPath === '/index.html')) {
      link.classList.add('active');
    } else {
      link.classList.remove('active');
    }
  });
}

// Populate the stock list with sectors and tickers
function populateStockList(sectors) {
  const stockListElement = document.getElementById('stockList');
  stockListElement.innerHTML = '';
  
  // Setup select/deselect buttons
  document.getElementById('selectAllBtn').addEventListener('click', () => {
    const allCheckboxes = document.querySelectorAll('.stock-checkbox');
    allCheckboxes.forEach(checkbox => {
      checkbox.checked = true;
      selectedTickers.add(checkbox.value);
    });
  });
  
  document.getElementById('deselectAllBtn').addEventListener('click', () => {
    const allCheckboxes = document.querySelectorAll('.stock-checkbox');
    allCheckboxes.forEach(checkbox => {
      checkbox.checked = false;
      selectedTickers.delete(checkbox.value);
    });
  });
  
  // Create sector groups
  sectors.forEach(sector => {
    const sectorGroup = document.createElement('div');
    sectorGroup.className = 'sector-group';
    
    // Sector header
    const sectorHeader = document.createElement('div');
    sectorHeader.className = 'sector-header';
    
    const sectorNameContainer = document.createElement('div');
    sectorNameContainer.className = 'sector-name';
    
    const sectorArrow = document.createElement('span');
    sectorArrow.className = 'sector-arrow';
    sectorArrow.textContent = '▶';
    sectorArrow.classList.add('open'); // Start with open sections
    
    const sectorNameText = document.createElement('span');
    sectorNameText.textContent = sector.sector;
    
    sectorNameContainer.appendChild(sectorArrow);
    sectorNameContainer.appendChild(sectorNameText);
    
    const sectorToggle = document.createElement('button');
    sectorToggle.className = 'sector-toggle';
    sectorToggle.textContent = 'Toggle All';
    
    sectorHeader.appendChild(sectorNameContainer);
    sectorHeader.appendChild(sectorToggle);
    sectorGroup.appendChild(sectorHeader);
    
    // Collapsible ticker grid
    const tickerGrid = document.createElement('div');
    tickerGrid.className = 'ticker-grid';
    
    // Create stock items in a grid
    sector.tickers.forEach(ticker => {
      const stockItem = document.createElement('div');
      stockItem.className = 'stock-item';
      
      const checkbox = document.createElement('input');
      checkbox.type = 'checkbox';
      checkbox.className = 'stock-checkbox';
      checkbox.value = ticker;
      checkbox.id = `ticker-${ticker}`;
      checkbox.addEventListener('change', () => {
        if (checkbox.checked) {
          selectedTickers.add(ticker);
        } else {
          selectedTickers.delete(ticker);
        }
      });
      
      const label = document.createElement('label');
      label.className = 'stock-ticker';
      label.textContent = ticker;
      label.htmlFor = `ticker-${ticker}`;
      
      stockItem.appendChild(checkbox);
      stockItem.appendChild(label);
      tickerGrid.appendChild(stockItem);
    });
    
    sectorGroup.appendChild(tickerGrid);
    stockListElement.appendChild(sectorGroup);
    
    // Toggle functionality
    sectorToggle.addEventListener('click', () => {
      const checkboxes = tickerGrid.querySelectorAll('.stock-checkbox');
      const allChecked = Array.from(checkboxes).every(cb => cb.checked);
      
      checkboxes.forEach(checkbox => {
        checkbox.checked = !allChecked;
        if (!allChecked) {
          selectedTickers.add(checkbox.value);
        } else {
          selectedTickers.delete(checkbox.value);
        }
      });
    });
    
    // Collapsible section
    sectorNameContainer.addEventListener('click', () => {
      sectorArrow.classList.toggle('open');
      tickerGrid.style.display = sectorArrow.classList.contains('open') ? 'grid' : 'none';
    });
  });
}

// Initialize charts
function initializeCharts() {
  const returnsCtx = document.getElementById('returnsChart').getContext('2d');
  const weightsCtx = document.getElementById('weightsChart').getContext('2d');
  const concentrationCtx = document.getElementById('concentrationChart').getContext('2d');
  
  // Configure Chart.js global defaults for dark theme
  Chart.defaults.color = '#b0b0b8';
  Chart.defaults.scale.grid.color = 'rgba(56, 56, 64, 0.5)';
  Chart.defaults.scale.grid.borderColor = 'rgba(56, 56, 64, 0.8)';
  
  // Returns chart configuration
  returnChart = new Chart(returnsCtx, {
    type: 'line',
    data: {
      datasets: [
        {
          label: 'OGD Portfolio',
          borderColor: '#3f88e2',
          backgroundColor: 'rgba(63, 136, 226, 0.1)',
          borderWidth: 1.5,
          pointRadius: 0, // Hide points completely
          tension: 0.1,
          data: []
        },
        {
          label: 'Equal Weight',
          borderColor: '#4caf50',
          backgroundColor: 'rgba(76, 175, 80, 0.1)',
          borderWidth: 1.5,
          pointRadius: 0, // Hide points completely
          tension: 0.1,
          data: []
        },
        {
          label: 'Random Portfolio',
          borderColor: '#e2b53f',
          backgroundColor: 'rgba(226, 181, 63, 0.1)',
          borderWidth: 1.5,
          pointRadius: 0, // Hide points completely
          tension: 0.1,
          data: []
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: 'index',
        intersect: false
      },
      scales: {
        x: {
          type: 'time',
          time: {
            unit: 'month',
            displayFormats: {
              month: 'MMM yyyy'
            }
          },
          title: {
            display: true,
            text: 'Date'
          }
        },
        y: {
          title: {
            display: true,
            text: 'Cumulative Return'
          },
          beginAtZero: false
        }
      },
      plugins: {
        legend: {
          position: 'top',
          labels: {
            usePointStyle: true,
            padding: 15
          }
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              const label = context.dataset.label || '';
              const value = ((context.parsed.y - 1) * 100).toFixed(2);
              return `${label}: ${value}%`;
            }
          }
        }
      }
    }
  });
  
  // Weights chart configuration
  weightChart = new Chart(weightsCtx, {
    type: 'line',
    data: {
      datasets: []
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: 'index',
        intersect: false
      },
      scales: {
        x: {
          type: 'time',
          time: {
            unit: 'month',
            displayFormats: {
              month: 'MMM yyyy'
            }
          },
          title: {
            display: true,
            text: 'Date'
          }
        },
        y: {
          title: {
            display: true,
            text: 'Asset Weight'
          },
          min: 0,
          ticks: {
            callback: function(value) {
              return (value * 100).toFixed(0) + '%';
            }
          }
        }
      },
      plugins: {
        legend: {
          position: 'top',
          labels: {
            usePointStyle: true,
            padding: 15
          }
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              const label = context.dataset.label || '';
              const value = (context.parsed.y * 100).toFixed(2);
              return `${label}: ${value}%`;
            }
          }
        }
      },
      elements: {
        line: {
          borderWidth: 1  // Default to thin lines
        },
        point: {
          radius: 0,     // Hide points by default
          hoverRadius: 3  // Show on hover
        }
      }
    }
  });
  
  // Concentration chart configuration
  concentrationChart = new Chart(concentrationCtx, {
    type: 'line',
    data: {
      datasets: [
        {
          label: 'Effective Number of Positions',
          borderColor: '#3f88e2',
          backgroundColor: 'rgba(63, 136, 226, 0.1)',
          borderWidth: 1.5,
          pointRadius: 0,  // Hide points
          pointHoverRadius: 3,  // Show points on hover
          tension: 0.1,
          data: []
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: 'index',
        intersect: false
      },
      scales: {
        x: {
          type: 'time',
          time: {
            unit: 'month',
            displayFormats: {
              month: 'MMM yyyy'
            }
          },
          title: {
            display: true,
            text: 'Date'
          }
        },
        y: {
          title: {
            display: true,
            text: 'ENP'
          },
          beginAtZero: false
        }
      },
      plugins: {
        legend: {
          position: 'top',
          labels: {
            usePointStyle: true,
            padding: 15
          }
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              const label = context.dataset.label || '';
              const value = context.parsed.y.toFixed(2);
              return `${label}: ${value}`;
            }
          }
        },
        annotation: {
          annotations: []
        }
      },
      elements: {
        line: {
          borderWidth: 1.5  // Default to thin lines
        },
        point: {
          radius: 0,      // Hide points by default
          hoverRadius: 3   // Show on hover
        }
      }
    }
  });
}

// Run the optimization when the user clicks the button
async function runOptimization() {
  // Get form values
  const startDate = document.getElementById('startDate').value;
  const endDate = document.getElementById('endDate').value;
  const learningRate = document.getElementById('learningRate').value;
  const windowSize = document.getElementById('windowSize').value;
  const alphaSortino = document.getElementById('alphaSortino').value;
  const alphaMaxDrawdown = document.getElementById('alphaMaxDrawdown').value;
  const alphaTurnover = document.getElementById('alphaTurnover').value;
  const alphaConcentration = document.getElementById('alphaConcentration').value;
  const enpMin = document.getElementById('enpMin').value;
  const enpMax = document.getElementById('enpMax').value;
  
  // Convert selected tickers to a comma-separated string
  const tickers = Array.from(selectedTickers).join(',');
  
  // Show loading overlay
  const loadingOverlay = document.getElementById('loadingOverlay');
  loadingOverlay.style.display = 'flex';
  
  // Backup the current state of the page for rollback if needed
  const statsRow = document.querySelector('.stats-row');
  const originalStatsContent = statsRow ? statsRow.innerHTML : '';
  
  // Create request data
  const requestData = {
    start_date: startDate,
    end_date: endDate,
    learning_rate: parseFloat(learningRate),
    window_size: parseInt(windowSize),
    alpha_sortino: parseFloat(alphaSortino),
    alpha_max_drawdown: parseFloat(alphaMaxDrawdown),
    alpha_turnover: parseFloat(alphaTurnover),
    alpha_concentration: parseFloat(alphaConcentration),
    enp_min: parseFloat(enpMin),
    enp_max: parseFloat(enpMax),
    tickers: tickers
  };
  
  try {
    // Send the request to the server
    const response = await fetch('/api/run_optimization', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(requestData)
    });
    
    // Check if the response is ok (status code 200-299)
    if (!response.ok) {
      throw new Error(`HTTP error! Status: ${response.status}`);
    }
    
    // Parse the response
    const data = await response.json();
    
    // Hide loading overlay
    loadingOverlay.style.display = 'none';
    
    // Check if there was an error
    if (data.error) {
      alert(`Optimization failed: ${data.error}`);
      console.error('Optimization error:', data.error);
      return;
    }
    
    // Update the charts with the new data
    updateCharts(data.cumulative_returns, data.weights, data.concentration, enpMin, enpMax);
    
    // Update the metrics grid
    updateMetrics(data.metrics);
    
  } catch (error) {
    console.error('Error during optimization:', error);
    
    // Hide loading overlay
    loadingOverlay.style.display = 'none';
    
    // Restore original stats content if there was an error
    if (statsRow && originalStatsContent) {
      statsRow.innerHTML = originalStatsContent;
    }
    
    // Show error message
    alert(`Optimization failed: ${error.message}. Check the console for more details.`);
    
    // Run fake simulation for demo purposes
    if (confirm('Would you like to see a demo simulation instead?')) {
      runFakeSimulation();
    }
  }
}

// Update charts with data from API response
function updateCharts(returnsData, weightsData, concentrationData, enpMin, enpMax) {
  if (!returnsData || !returnChart) {
    console.error('Returns data or chart not available');
    return;
  }
  
  // Filter out any data points with null values
  const ogdData = returnsData.ogd.filter(d => d.value !== null);
  const equalWeightData = returnsData.equal_weight.filter(d => d.value !== null);
  const randomData = returnsData.random.filter(d => d.value !== null);
  
  // Update returns chart
  returnChart.data.datasets[0].data = ogdData.map(d => ({ x: d.date, y: d.value }));
  returnChart.data.datasets[1].data = equalWeightData.map(d => ({ x: d.date, y: d.value }));
  returnChart.data.datasets[2].data = randomData.map(d => ({ x: d.date, y: d.value }));
  returnChart.update();
  
  // Update weights chart if data is available
  if (weightsData && weightsData.length > 0 && weightChart) {
    // Get unique tickers from all dates
    const allTickers = new Set();
    weightsData.forEach(day => {
      Object.keys(day.weights).forEach(ticker => allTickers.add(ticker));
    });
    
    // Create datasets for each ticker
    const datasets = Array.from(allTickers).map(ticker => {
      // Get a color for this ticker
      const color = getColor(ticker);
      
      // Create data points for each date
      const data = weightsData.map(day => ({
        x: day.date,
        y: day.weights[ticker] || 0
      }));
      
      // Filter out any null values
      const validData = data.filter(d => d.y !== null);
      
      return {
        label: ticker,
        data: validData,
        backgroundColor: color.replace('0.7', '0.2'),  // Make fill more transparent
        borderColor: color,
        fill: true,
        tension: 0.1,
        borderWidth: 1,  // Make line thinner
        pointRadius: 0,  // Hide points completely
        pointHoverRadius: 3  // Show points on hover
      };
    });
    
    // Update the chart
    weightChart.data.datasets = datasets;
    weightChart.update();
  }
  
  // Update concentration chart if data is available
  if (concentrationData && concentrationData.enp && concentrationData.enp.length > 0 && concentrationChart) {
    // Filter out any data points with null values
    const enpData = concentrationData.enp.filter(d => d.value !== null);
    
    // Update the chart
    concentrationChart.data.datasets[0].data = enpData.map(d => ({ x: d.date, y: d.value }));
    concentrationChart.data.datasets[0].borderWidth = 1.5;  // Make line thinner
    concentrationChart.data.datasets[0].pointRadius = 0;  // Hide points completely
    concentrationChart.data.datasets[0].pointHoverRadius = 3;  // Show points on hover
    
    // Add min and max ENP target lines
    if (concentrationChart.options.plugins.annotation.annotations.length === 0) {
      concentrationChart.options.plugins.annotation.annotations.push({
        type: 'line',
        borderColor: 'rgba(244, 67, 54, 0.5)',
        borderWidth: 1,
        borderDash: [6, 6],
        label: {
          enabled: true,
          content: 'Min Target',
          position: 'start'
        },
        scaleID: 'y',
        value: enpMin || 5.0
      });
      
      concentrationChart.options.plugins.annotation.annotations.push({
        type: 'line',
        borderColor: 'rgba(76, 175, 80, 0.5)',
        borderWidth: 1,
        borderDash: [6, 6],
        label: {
          enabled: true,
          content: 'Max Target',
          position: 'start'
        },
        scaleID: 'y',
        value: enpMax || 20.0
      });
    } else {
      // Update existing annotation values
      concentrationChart.options.plugins.annotation.annotations[0].value = enpMin || 5.0;
      concentrationChart.options.plugins.annotation.annotations[1].value = enpMax || 20.0;
    }
    
    concentrationChart.update();
  }
}

// Update performance metrics for all three portfolios
function updateMetrics(metrics) {
  // Format values for display
  const formatValue = (value, format = 'decimal') => {
    if (value === undefined || value === null) return '-';
    
    if (format === 'percent') {
      // For cumulative returns, we need to subtract 1 first to get the percent change
      if (format === 'percent' && value > 1) {
        // This is a cumulative return (started at 1)
        return ((value - 1) * 100).toFixed(2) + '%';
      }
      return (value * 100).toFixed(2) + '%';
    } else if (format === 'decimal') {
      return value.toFixed(2);
    }
    return value;
  };

  // Make sure metrics objects exist to avoid errors
  if (!metrics) {
    console.error('No metrics data provided to updateMetrics');
    metrics = {
      ogd: { sharpe: 0, max_drawdown: 0, cumulative_return: 1, capm_alpha: 0, ff3_alpha: 0 },
      equal_weight: { sharpe: 0, max_drawdown: 0, cumulative_return: 1, capm_alpha: 0, ff3_alpha: 0 },
      random: { sharpe: 0, max_drawdown: 0, cumulative_return: 1, capm_alpha: 0, ff3_alpha: 0 }
    };
  }
  
  // Initialize metric objects if missing
  if (!metrics.ogd) metrics.ogd = {};
  if (!metrics.equal_weight) metrics.equal_weight = {};
  if (!metrics.random) metrics.random = {};

  const statsRow = document.querySelector('.stats-row');
  if (!statsRow) {
    console.error('Stats row element not found');
    return;
  }
  
  // Clear the grid
  statsRow.innerHTML = '';
  
  // Create strategy titles (column headers)
  const strategies = [
    { id: 'ogd', name: 'OGD Portfolio', class: 'ogd-strategy' },
    { id: 'equal_weight', name: 'Equal Weight', class: 'equal-weight-strategy' },
    { id: 'random', name: 'Random Portfolio', class: 'random-strategy' }
  ];
  
  // Define metrics to display (row headers)
  const metricTypes = [
    { id: 'sharpe', name: 'Sharpe Ratio', format: 'decimal' },
    { id: 'max_drawdown', name: 'Max Drawdown', format: 'percent' },
    { id: 'cumulative_return', name: 'Return', format: 'percent' },
    { id: 'capm_alpha', name: 'CAPM Alpha', format: 'percent' },
    { id: 'ff3_alpha', name: 'FF3 Alpha', format: 'percent' }
  ];
  
  // Add column headers (strategy names)
  // Empty cell for top-left corner
  const emptyHeader = document.createElement('div');
  emptyHeader.className = 'stat-card metric-label strategy-header';
  emptyHeader.innerHTML = '<div class="stat-value">Metrics</div>';
  statsRow.appendChild(emptyHeader);
  
  // Create strategy headers
  strategies.forEach(strategy => {
    const strategyHeader = document.createElement('div');
    strategyHeader.className = `stat-card strategy-header ${strategy.class}`;
    strategyHeader.innerHTML = `<div class="stat-value">${strategy.name}</div>`;
    statsRow.appendChild(strategyHeader);
  });
  
  // Add metric rows
  metricTypes.forEach(metric => {
    // Add row label
    const metricLabel = document.createElement('div');
    metricLabel.className = 'stat-card metric-label';
    metricLabel.innerHTML = `<div class="stat-value">${metric.name}</div>`;
    statsRow.appendChild(metricLabel);
    
    // Add values for each strategy
    strategies.forEach(strategy => {
      // Get the value safely
      const value = metrics[strategy.id][metric.id];
      let formattedValue = '-';
      
      // Only format the value if it's not null or undefined
      if (value !== null && value !== undefined) {
        // Handle cumulative return differently
        if (metric.id === 'cumulative_return') {
          formattedValue = ((value - 1) * 100).toFixed(2) + '%';
        } else if (metric.format === 'percent') {
          formattedValue = (value * 100).toFixed(2) + '%';
        } else {
          formattedValue = formatValue(value, metric.format);
        }
      }
      
      const metricCell = document.createElement('div');
      metricCell.className = `stat-card metric-row ${strategy.class}`;
      metricCell.setAttribute('data-metric', metric.name);
      metricCell.innerHTML = `<div class="stat-value">${formattedValue}</div>`;
      statsRow.appendChild(metricCell);
    });
  });
  
  // Log grid creation
  console.log('Metrics grid created with', strategies.length * metricTypes.length, 'cells');
  
  // Update legacy hidden elements
  try {
    // Legacy element ID updates for backward compatibility
    const ogdSharpe = document.getElementById('ogdSharpeRatio');
    if (ogdSharpe) ogdSharpe.textContent = formatValue(metrics.ogd.sharpe);
    
    const ogdDrawdown = document.getElementById('ogdMaxDrawdown');
    if (ogdDrawdown) ogdDrawdown.textContent = formatValue(metrics.ogd.max_drawdown, 'percent');
    
    const ogdReturn = document.getElementById('ogdReturn');
    if (ogdReturn && metrics.ogd.cumulative_return !== null) {
      ogdReturn.textContent = ((metrics.ogd.cumulative_return - 1) * 100).toFixed(2) + '%';
    } else if (ogdReturn) {
      ogdReturn.textContent = '-';
    }
    
    const ewSharpe = document.getElementById('ewSharpeRatio');
    if (ewSharpe) ewSharpe.textContent = formatValue(metrics.equal_weight.sharpe);
    
    const ewDrawdown = document.getElementById('ewMaxDrawdown');
    if (ewDrawdown) ewDrawdown.textContent = formatValue(metrics.equal_weight.max_drawdown, 'percent');
    
    const ewReturn = document.getElementById('ewReturn');
    if (ewReturn && metrics.equal_weight.cumulative_return !== null) {
      ewReturn.textContent = ((metrics.equal_weight.cumulative_return - 1) * 100).toFixed(2) + '%';
    } else if (ewReturn) {
      ewReturn.textContent = '-';
    }
    
    const randomSharpe = document.getElementById('randomSharpeRatio');
    if (randomSharpe) randomSharpe.textContent = formatValue(metrics.random.sharpe);
    
    const randomDrawdown = document.getElementById('randomMaxDrawdown');
    if (randomDrawdown) randomDrawdown.textContent = formatValue(metrics.random.max_drawdown, 'percent');
    
    const randomReturn = document.getElementById('randomReturn');
    if (randomReturn && metrics.random.cumulative_return !== null) {
      randomReturn.textContent = ((metrics.random.cumulative_return - 1) * 100).toFixed(2) + '%';
    } else if (randomReturn) {
      randomReturn.textContent = '-';
    }
    
    // Update factor alphas if available
    if (metrics.ogd.capm_alpha !== undefined) {
      const ogdCAPM = document.getElementById('ogdCAPMAlpha');
      if (ogdCAPM) ogdCAPM.textContent = formatValue(metrics.ogd.capm_alpha, 'percent');
      
      const ogdFF3 = document.getElementById('ogdFF3Alpha');
      if (ogdFF3) ogdFF3.textContent = formatValue(metrics.ogd.ff3_alpha, 'percent');
      
      const ewCAPM = document.getElementById('ewCAPMAlpha');
      if (ewCAPM) ewCAPM.textContent = formatValue(metrics.equal_weight.capm_alpha, 'percent');
      
      const ewFF3 = document.getElementById('ewFF3Alpha');
      if (ewFF3) ewFF3.textContent = formatValue(metrics.equal_weight.ff3_alpha, 'percent');
      
      const randomCAPM = document.getElementById('randomCAPMAlpha');
      if (randomCAPM) randomCAPM.textContent = formatValue(metrics.random.capm_alpha, 'percent');
      
      const randomFF3 = document.getElementById('randomFF3Alpha');
      if (randomFF3) randomFF3.textContent = formatValue(metrics.random.ff3_alpha, 'percent');
    }
  } catch (e) {
    console.warn('Error updating legacy elements:', e);
    // Continue execution even if legacy element update fails
  }
}

// Create a fake simulation for visual feedback during loading
function runFakeSimulation() {
  const loadingOverlay = document.getElementById('loadingOverlay');
  if (loadingOverlay.style.display === 'none') return;
  
  // Get ENP min/max values
  const enpMin = parseFloat(document.getElementById('enpMin').value) || 5.0;
  const enpMax = parseFloat(document.getElementById('enpMax').value) || 20.0;
  
  // Generate fake return data
  const dates = [];
  const today = new Date();
  for (let i = 365; i >= 0; i--) {
    const date = new Date(today);
    date.setDate(today.getDate() - i);
    dates.push(date.toISOString().split('T')[0]);
  }
  
  // Generate fake returns
  const generateFakeData = (volatility, bias = 0) => {
    let value = 1.0;
    return dates.map(date => {
      value *= (1 + (Math.random() - 0.45 + bias) * volatility);
      return { date, value };
    });
  };
  
  const fakeReturns = {
    ogd: generateFakeData(0.015, 0.02),
    equal_weight: generateFakeData(0.01, 0.01),
    random: generateFakeData(0.02, 0)
  };
  
  // Generate fake weights
  const fakeWeights = [];
  const fakeTickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'JNJ', 'V'];
  
  dates.forEach(date => {
    const weights = {};
    let totalWeight = 0;
    
    fakeTickers.forEach(ticker => {
      // Random weight but normalized later
      weights[ticker] = Math.random();
      totalWeight += weights[ticker];
    });
    
    // Normalize weights to sum to 1
    Object.keys(weights).forEach(ticker => {
      weights[ticker] = weights[ticker] / totalWeight;
    });
    
    fakeWeights.push({ date, weights });
  });
  
  // Generate fake concentration data
  const fakeConcentration = {
    enp: dates.map(date => {
      // Random ENP value that fluctuates between min and max with some noise
      const range = enpMax - enpMin;
      const center = enpMin + range / 2;
      const variation = range / 3;
      const value = center + (Math.random() * 2 - 1) * variation;
      return { date, value };
    }),
    hhi: dates.map(date => {
      return { date, value: Math.random() * 0.3 + 0.1 };
    })
  };
  
  // Update charts with fake data
  updateCharts(fakeReturns, fakeWeights, fakeConcentration, enpMin, enpMax);
  
  // Update metrics with fake data
  updateMetrics({
    ogd: {
      sharpe: 1.2 + Math.random() * 0.5,
      max_drawdown: 0.12 + Math.random() * 0.1,
      cumulative_return: 1.3 + Math.random() * 0.5,
      capm_alpha: 2.5 + Math.random() * 1.0, // Annual % 
      ff3_alpha: 1.8 + Math.random() * 0.8   // Annual %
    },
    equal_weight: {
      sharpe: 0.9 + Math.random() * 0.3,
      max_drawdown: 0.15 + Math.random() * 0.1,
      cumulative_return: 1.2 + Math.random() * 0.3,
      capm_alpha: 1.2 + Math.random() * 0.5,
      ff3_alpha: 0.8 + Math.random() * 0.4
    },
    random: {
      sharpe: 0.6 + Math.random() * 0.3,
      max_drawdown: 0.2 + Math.random() * 0.15,
      cumulative_return: 1.1 + Math.random() * 0.2,
      capm_alpha: 0.4 + Math.random() * 0.6,
      ff3_alpha: 0.2 + Math.random() * 0.4
    }
  });
}

// Generate a color from a ticker symbol
function getColor(ticker) {
  // Use cached color if already assigned
  if (colorMap[ticker]) {
    return colorMap[ticker];
  }
  
  // Predefined colors for common tickers
  const predefinedColors = {
    'AAPL': 'rgba(63, 136, 226, 0.7)',  // Blue
    'MSFT': 'rgba(76, 175, 80, 0.7)',   // Green
    'GOOGL': 'rgba(226, 181, 63, 0.7)', // Yellow
    'AMZN': 'rgba(226, 77, 63, 0.7)',   // Red
    'META': 'rgba(156, 39, 176, 0.7)',  // Purple
    'TSLA': 'rgba(0, 188, 212, 0.7)',   // Cyan
    'NVDA': 'rgba(255, 235, 59, 0.7)',  // Bright Yellow
    'JPM': 'rgba(121, 85, 72, 0.7)',    // Brown
    'V': 'rgba(96, 125, 139, 0.7)',     // Blue Grey
    'JNJ': 'rgba(233, 30, 99, 0.7)'     // Pink
  };
  
  if (predefinedColors[ticker]) {
    colorMap[ticker] = predefinedColors[ticker];
    return colorMap[ticker];
  }
  
  // Generate a pseudo-random color based on the ticker string
  let hash = 0;
  for (let i = 0; i < ticker.length; i++) {
    hash = ticker.charCodeAt(i) + ((hash << 5) - hash);
  }
  
  // Convert to RGB color with proper range to ensure readability
  const r = (hash & 0xFF) % 200 + 55;  // Range: 55-255 (avoid too dark)
  const g = ((hash >> 8) & 0xFF) % 200 + 55;
  const b = ((hash >> 16) & 0xFF) % 200 + 55;
  
  // Convert to rgba color string with 70% opacity for better visibility
  const color = `rgba(${r}, ${g}, ${b}, 0.7)`;
  
  // Cache the color
  colorMap[ticker] = color;
  return color;
}

// Initialize the application
async function initialize() {
  try {
    // Set active navigation link
    setActiveNavLink();
    
    // Initialize charts
    initializeCharts();
    
    // Set default dates to match the data range
    document.getElementById('startDate').value = '2007-01-01'; // Jan 1, 2007
    document.getElementById('endDate').value = '2025-04-01';   // Apr 1, 2025
    
    // Fetch tickers and populate stock list
    sectorData = await fetchTickersBySector();
    populateStockList(sectorData);
    
    // Add event listener to run button
    document.getElementById('runButton').addEventListener('click', runOptimization);
    
    // Create initial metrics grid
    updateMetrics({
      ogd: {
        sharpe: 1.2,
        max_drawdown: 0.12,
        cumulative_return: 1.3,
        capm_alpha: 2.5,
        ff3_alpha: 1.8
      },
      equal_weight: {
        sharpe: 0.9,
        max_drawdown: 0.15,
        cumulative_return: 1.2,
        capm_alpha: 1.2,
        ff3_alpha: 0.8
      },
      random: {
        sharpe: 0.6,
        max_drawdown: 0.2,
        cumulative_return: 1.1,
        capm_alpha: 0.4,
        ff3_alpha: 0.2
      }
    });
    
    // Run fake simulation for initial visual
    runFakeSimulation();
    
  } catch (error) {
    console.error('Error initializing application:', error);
  }
}

// Run the initialization when the document is loaded
document.addEventListener('DOMContentLoaded', initialize); 