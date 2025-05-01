// Register the Chart.js annotation plugin if available
try {
  // In Chart.js 3.x, the plugin is imported separately but needs to be registered
  // In some bundled versions, it might be automatically available
  const hasAnnotationPlugin = 
    typeof window.ChartAnnotation !== 'undefined' || 
    (Chart && Chart.Annotation) || 
    (Chart && Chart.registry && Chart.registry.plugins && Chart.registry.plugins.get('annotation'));
  
  console.log('Chart.js version:', Chart.version);
  console.log('Annotation plugin available:', hasAnnotationPlugin);
  
  if (typeof window.ChartAnnotation !== 'undefined') {
    Chart.register(window.ChartAnnotation);
    console.log('Registered external ChartAnnotation plugin');
  } else {
    console.log('Using bundled annotation plugin or proceeding without it');
  }
} catch (error) {
  console.error('Error during Chart.js plugin setup:', error);
}

// Education page JavaScript

document.addEventListener('DOMContentLoaded', function() {
  console.log('Education page loaded');
  
  // Mark the current page in the navigation
  const currentPath = window.location.pathname;
  const navLinks = document.querySelectorAll('.nav-link');
  
  navLinks.forEach(link => {
    const linkPath = link.getAttribute('href');
    if (currentPath.endsWith(linkPath) || 
        (currentPath.endsWith('/') && linkPath === '/index.html')) {
      link.classList.add('active');
    } else {
      link.classList.remove('active');
    }
  });

  // Initialize the Sortino ratio demonstration
  initializeSortinoDemo();
  
  // Initialize the Maximum Drawdown demonstration
  initializeMDDDemo();
  
  // Initialize the Turnover Ratio demonstration
  initializeTurnoverDemo();
  
  // Initialize the Concentration Penalty demonstration
  initializeConcentrationDemo();
});

// Utility function to calculate cumulative returns
function calculateCumulativeReturns(returns) {
  let cumulative = 1.0;
  return returns.map(ret => {
    cumulative *= (1 + ret);
    return cumulative;
  });
}

// Utility function to calculate drawdowns from cumulative returns
function calculateDrawdowns(cumulativeReturns) {
  let peak = cumulativeReturns[0];
  return cumulativeReturns.map(cr => {
    peak = Math.max(peak, cr);
    return (peak - cr) / peak; // Drawdown as a percentage of the peak
  });
}

// Utility function to calculate maximum drawdown
function calculateMaxDrawdown(cumulativeReturns) {
  let peak = cumulativeReturns[0];
  let maxDrawdown = 0;
  
  cumulativeReturns.forEach(cr => {
    peak = Math.max(peak, cr);
    const drawdown = (peak - cr) / peak;
    maxDrawdown = Math.max(maxDrawdown, drawdown);
  });
  
  return maxDrawdown;
}

// Utility function to find drawdown periods
function findDrawdownPeriods(cumulativeReturns, threshold = 0.05) {
  const periods = [];
  let inDrawdown = false;
  let startIndex = 0;
  let peak = cumulativeReturns[0];
  let peakIndex = 0;
  
  cumulativeReturns.forEach((cr, i) => {
    if (cr > peak) {
      peak = cr;
      peakIndex = i;
      
      // If we were in a drawdown, it's now over
      if (inDrawdown) {
        periods.push({
          start: startIndex,
          end: i - 1,
          peakIndex: peakIndex
        });
        inDrawdown = false;
      }
    } else {
      const drawdown = (peak - cr) / peak;
      
      // Start tracking a new drawdown once it exceeds the threshold
      if (!inDrawdown && drawdown >= threshold) {
        inDrawdown = true;
        startIndex = i;
      }
    }
  });
  
  // If we're still in a drawdown at the end, add the final period
  if (inDrawdown) {
    periods.push({
      start: startIndex,
      end: cumulativeReturns.length - 1,
      peakIndex: peakIndex
    });
  }
  
  return periods;
}

// Utility function to calculate Sortino ratio
function calculateSortinoRatio(returns, riskFreeRate, windowSize) {
  if (returns.length < windowSize) {
    return 0;
  }
  
  // Calculate excess returns
  const excessReturns = returns.map(r => r - riskFreeRate);
  
  // Calculate average excess return
  const avgExcessReturn = excessReturns.reduce((a, b) => a + b, 0) / excessReturns.length;
  
  // Calculate downside deviation (only consider negative excess returns)
  const negativeExcessReturns = excessReturns.filter(r => r < 0);
  if (negativeExcessReturns.length === 0) {
    return 3; // Arbitrary high value when there are no negative returns
  }
  
  const downsideDeviation = Math.sqrt(
    negativeExcessReturns.reduce((sum, r) => sum + r * r, 0) / negativeExcessReturns.length
  );
  
  // Small constant to avoid division by zero
  const epsilon = 0.0001;
  
  // Calculate Sortino ratio
  return avgExcessReturn / (downsideDeviation + epsilon);
}

// Utility function to calculate Sharpe ratio
function calculateSharpeRatio(returns, riskFreeRate, windowSize) {
  if (returns.length < windowSize) {
    return 0;
  }
  
  // Calculate excess returns
  const excessReturns = returns.map(r => r - riskFreeRate);
  
  // Calculate average excess return
  const avgExcessReturn = excessReturns.reduce((a, b) => a + b, 0) / excessReturns.length;
  
  // Calculate standard deviation of returns (total volatility)
  const meanReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
  const variance = returns.reduce((sum, r) => sum + Math.pow(r - meanReturn, 2), 0) / returns.length;
  const stdDev = Math.sqrt(variance);
  
  // Small constant to avoid division by zero
  const epsilon = 0.0001;
  
  // Calculate Sharpe ratio
  return avgExcessReturn / (stdDev + epsilon);
}

// Generate synthetic data for assets with different risk-return profiles
function generateSyntheticData(days = 120) {
  const riskFreeRate = 0.0001; // Daily risk-free rate (approximately 2.5% annually)
  const dates = [];
  const today = new Date();
  
  // Generate dates for the past 4 months
  for (let i = days; i >= 0; i--) {
    const date = new Date(today);
    date.setDate(today.getDate() - i);
    dates.push(date);
  }
  
  // Asset A: Highly volatile with spikey behavior (dramatic ups and downs)
  const assetAReturns = Array(days + 1).fill(0).map((_, i) => {
    // Base pattern with occasional sharp spikes both positive and negative
    let baseReturn = 0.0005;
    
    // Add spikes every ~10 days
    if (i % 10 === 0) {
      // Alternate between positive and negative spikes
      return (i % 20 === 0) ? 0.025 : -0.02;
    } else if (i % 10 === 1) {
      // Follow-up correction
      return (i % 20 === 1) ? -0.01 : 0.008;
    }
    
    // Regular volatility on other days
    return baseReturn + (Math.random() - 0.5) * 0.012;
  });
  
  // Asset B: Smooth exponential growth with minimal downside
  const assetBReturns = Array(days + 1).fill(0).map((_, i) => {
    // Gradually increasing return rate to create exponential curve
    const growthFactor = 1 + (i / (days * 5));
    const baseReturn = 0.001 * growthFactor;
    
    // Add very limited downside with 20% chance of small negative returns
    const rand = Math.random();
    if (rand < 0.2) {
      return -0.0015; // Small controlled negative returns
    } else {
      return baseReturn + (rand * 0.005); // Positive returns with upward trend
    }
  });
  
  // Asset C: Nearly linear growth with minimal volatility
  const assetCReturns = Array(days + 1).fill(0).map(() => {
    // Very consistent returns with minimal variation
    return 0.0003 + (Math.random() - 0.5) * 0.0006;
  });
  
  // Calculate cumulative returns
  const assetACumulative = calculateCumulativeReturns(assetAReturns);
  const assetBCumulative = calculateCumulativeReturns(assetBReturns);
  const assetCCumulative = calculateCumulativeReturns(assetCReturns);
  
  // Calculate rolling Sortino ratios with a 10-day window (shorter window for more variation)
  const windowSize = 10;
  const assetASortino = [];
  const assetBSortino = [];
  const assetCSortino = [];
  
  // Calculate rolling Sharpe ratios with the same window
  const assetASharpe = [];
  const assetBSharpe = [];
  const assetCSharpe = [];
  
  for (let i = 0; i <= days; i++) {
    const startIdx = Math.max(0, i - windowSize + 1);
    const windowA = assetAReturns.slice(startIdx, i + 1);
    const windowB = assetBReturns.slice(startIdx, i + 1);
    const windowC = assetCReturns.slice(startIdx, i + 1);
    
    assetASortino.push(calculateSortinoRatio(windowA, riskFreeRate, windowSize));
    assetBSortino.push(calculateSortinoRatio(windowB, riskFreeRate, windowSize));
    assetCSortino.push(calculateSortinoRatio(windowC, riskFreeRate, windowSize));
    
    assetASharpe.push(calculateSharpeRatio(windowA, riskFreeRate, windowSize));
    assetBSharpe.push(calculateSharpeRatio(windowB, riskFreeRate, windowSize));
    assetCSharpe.push(calculateSharpeRatio(windowC, riskFreeRate, windowSize));
  }
  
  return {
    dates,
    returns: {
      assetA: assetAReturns,
      assetB: assetBReturns,
      assetC: assetCReturns,
      assetACumulative,
      assetBCumulative,
      assetCCumulative
    },
    sortinoRatios: {
      assetA: assetASortino,
      assetB: assetBSortino,
      assetC: assetCSortino
    },
    sharpeRatios: {
      assetA: assetASharpe,
      assetB: assetBSharpe,
      assetC: assetCSharpe
    }
  };
}

// Initialize the Sortino ratio demonstration
function initializeSortinoDemo() {
  // Generate synthetic data
  const data = generateSyntheticData();
  
  // Format dates for chart display
  const formattedDates = data.dates.map(d => {
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  });
  
  // Create returns chart
  const returnsCtx = document.getElementById('returnsChart').getContext('2d');
  new Chart(returnsCtx, {
    type: 'line',
    data: {
      labels: formattedDates,
      datasets: [
        {
          label: 'Asset A',
          data: data.returns.assetACumulative.map(val => (val - 1) * 100), // Convert to percentage
          borderColor: 'rgba(255, 99, 132, 1)',
          backgroundColor: 'rgba(255, 99, 132, 0.1)',
          borderWidth: 1,
          pointRadius: 0.5,
          pointHoverRadius: 3,
          tension: 0.1
        },
        {
          label: 'Asset B',
          data: data.returns.assetBCumulative.map(val => (val - 1) * 100),
          borderColor: 'rgba(54, 162, 235, 1)',
          backgroundColor: 'rgba(54, 162, 235, 0.1)',
          borderWidth: 1,
          pointRadius: 0.5,
          pointHoverRadius: 3,
          tension: 0.1
        },
        {
          label: 'Asset C',
          data: data.returns.assetCCumulative.map(val => (val - 1) * 100),
          borderColor: 'rgba(75, 192, 192, 1)',
          backgroundColor: 'rgba(75, 192, 192, 0.1)',
          borderWidth: 1,
          pointRadius: 0.5,
          pointHoverRadius: 3,
          tension: 0.1
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              return `${context.dataset.label}: ${context.raw.toFixed(2)}%`;
            }
          }
        }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: 'Date'
          },
          ticks: {
            maxTicksLimit: 10
          }
        },
        y: {
          title: {
            display: true,
            text: 'Cumulative Return (%)'
          }
        }
      }
    }
  });
  
  // Create Sharpe ratio chart
  const sharpeCtx = document.getElementById('sharpeChart').getContext('2d');
  new Chart(sharpeCtx, {
    type: 'line',
    data: {
      labels: formattedDates,
      datasets: [
        {
          label: 'Asset A',
          data: data.sharpeRatios.assetA,
          borderColor: 'rgba(255, 99, 132, 1)',
          backgroundColor: 'rgba(255, 99, 132, 0.1)',
          borderWidth: 1,
          pointRadius: 0.5,
          pointHoverRadius: 3,
          tension: 0.1
        },
        {
          label: 'Asset B',
          data: data.sharpeRatios.assetB,
          borderColor: 'rgba(54, 162, 235, 1)',
          backgroundColor: 'rgba(54, 162, 235, 0.1)',
          borderWidth: 1,
          pointRadius: 0.5,
          pointHoverRadius: 3,
          tension: 0.1
        },
        {
          label: 'Asset C',
          data: data.sharpeRatios.assetC,
          borderColor: 'rgba(75, 192, 192, 1)',
          backgroundColor: 'rgba(75, 192, 192, 0.1)',
          borderWidth: 1,
          pointRadius: 0.5,
          pointHoverRadius: 3,
          tension: 0.1
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              return `${context.dataset.label}: ${context.raw.toFixed(2)}`;
            }
          }
        }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: 'Date'
          },
          ticks: {
            maxTicksLimit: 10
          }
        },
        y: {
          title: {
            display: true,
            text: 'Sharpe Ratio (10-day)'
          },
          min: -2,
          max: 6
        }
      }
    }
  });
  
  // Create Sortino ratio chart
  const sortinoCtx = document.getElementById('sortinoChart').getContext('2d');
  new Chart(sortinoCtx, {
    type: 'line',
    data: {
      labels: formattedDates,
      datasets: [
        {
          label: 'Asset A',
          data: data.sortinoRatios.assetA,
          borderColor: 'rgba(255, 99, 132, 1)',
          backgroundColor: 'rgba(255, 99, 132, 0.1)',
          borderWidth: 1,
          pointRadius: 0.5,
          pointHoverRadius: 3,
          tension: 0.1
        },
        {
          label: 'Asset B',
          data: data.sortinoRatios.assetB,
          borderColor: 'rgba(54, 162, 235, 1)',
          backgroundColor: 'rgba(54, 162, 235, 0.1)',
          borderWidth: 1,
          pointRadius: 0.5,
          pointHoverRadius: 3,
          tension: 0.1
        },
        {
          label: 'Asset C',
          data: data.sortinoRatios.assetC,
          borderColor: 'rgba(75, 192, 192, 1)',
          backgroundColor: 'rgba(75, 192, 192, 0.1)',
          borderWidth: 1,
          pointRadius: 0.5,
          pointHoverRadius: 3,
          tension: 0.1
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              return `${context.dataset.label}: ${context.raw.toFixed(2)}`;
            }
          }
        }
      },
      scales: {
        x: {
          title: {
            display: true,
            text: 'Date'
          },
          ticks: {
            maxTicksLimit: 10
          }
        },
        y: {
          title: {
            display: true,
            text: 'Sortino Ratio (10-day)'
          },
          min: -2,
          max: 10
        }
      }
    }
  });
}

// Generate synthetic data for assets with different drawdown characteristics
function generateMDDSyntheticData(days = 120) {
  const dates = [];
  const today = new Date();
  
  // Generate dates for the past 4 months
  for (let i = days; i >= 0; i--) {
    const date = new Date(today);
    date.setDate(today.getDate() - i);
    dates.push(date);
  }
  
  // Asset D: Steady growth with one major correction
  const assetDReturns = Array(days + 1).fill(0).map((_, i) => {
    // Generally positive returns
    let baseReturn = 0.0010;
    
    // Add one major correction around day 50-65
    if (i >= 50 && i < 65) {
      // Sharp decline
      return (i === 50) ? -0.06 : -0.008;
    } else if (i >= 65 && i < 80) {
      // Recovery period with strong returns
      return 0.005;
    }
    
    // Regular positive drift with some noise
    return baseReturn + (Math.random() - 0.45) * 0.003;
  });
  
  // Asset E: Volatile with frequent smaller drawdowns
  const assetEReturns = Array(days + 1).fill(0).map((_, i) => {
    // Base pattern with frequent corrections
    let baseReturn = 0.0015;
    
    // Cyclical pattern: every 15-20 days a correction
    if (i % 20 > 15) {
      return -0.01 - Math.random() * 0.005;
    } else {
      return baseReturn + (Math.random() - 0.3) * 0.004;
    }
  });
  
  // Asset F: Declining trend with recovery
  const assetFReturns = Array(days + 1).fill(0).map((_, i) => {
    if (i < 60) {
      // Gradual decline in first half
      return -0.002 - Math.random() * 0.003;
    } else if (i < 80) {
      // Bottom formation
      return -0.0005 + (Math.random() - 0.5) * 0.002;
    } else {
      // Recovery in last part
      return 0.003 + Math.random() * 0.004;
    }
  });
  
  // Calculate cumulative returns
  const assetDCumulative = calculateCumulativeReturns(assetDReturns);
  const assetECumulative = calculateCumulativeReturns(assetEReturns);
  const assetFCumulative = calculateCumulativeReturns(assetFReturns);
  
  // Calculate drawdowns
  const assetDDrawdowns = calculateDrawdowns(assetDCumulative);
  const assetEDrawdowns = calculateDrawdowns(assetECumulative);
  const assetFDrawdowns = calculateDrawdowns(assetFCumulative);
  
  // Find significant drawdown periods
  const assetDPeriods = findDrawdownPeriods(assetDCumulative);
  const assetEPeriods = findDrawdownPeriods(assetECumulative);
  const assetFPeriods = findDrawdownPeriods(assetFCumulative);
  
  return {
    dates,
    returns: {
      assetD: assetDReturns,
      assetE: assetEReturns,
      assetF: assetFReturns
    },
    cumulative: {
      assetD: assetDCumulative,
      assetE: assetECumulative,
      assetF: assetFCumulative
    },
    drawdowns: {
      assetD: assetDDrawdowns,
      assetE: assetEDrawdowns,
      assetF: assetFDrawdowns
    },
    periods: {
      assetD: assetDPeriods,
      assetE: assetEPeriods,
      assetF: assetFPeriods
    },
    maxDrawdowns: {
      assetD: calculateMaxDrawdown(assetDCumulative),
      assetE: calculateMaxDrawdown(assetECumulative),
      assetF: calculateMaxDrawdown(assetFCumulative)
    }
  };
}

// Initialize the Maximum Drawdown demonstration charts
function initializeMDDDemo() {
  console.log('Initializing Maximum Drawdown demo');
  const mddReturnsChartCtx = document.getElementById('mddReturnsChart');
  const drawdownChartCtx = document.getElementById('drawdownChart');
  
  if (!mddReturnsChartCtx || !drawdownChartCtx) {
    console.error('Chart canvas elements not found for MDD demo');
    return;
  }
  
  // Generate synthetic data for MDD demo
  const syntheticData = generateMDDSyntheticData();
  
  // Create plugin for highlighting drawdown periods
  const drawdownPeriodPlugin = {
    id: 'drawdownPeriod',
    beforeDraw: (chart) => {
      const {ctx, chartArea, scales} = chart;
      
      if (!chartArea) {
        return;
      }
      
      // Get the datasets
      const datasets = chart.data.datasets;
      for (let i = 0; i < datasets.length; i++) {
        const dataset = datasets[i];
        // Skip if it's not a dataset we want to highlight periods for
        if (!dataset.drawdownPeriods) continue;
        
        const periods = dataset.drawdownPeriods;
        
        // Draw each drawdown period
        periods.forEach(period => {
          const startX = scales.x.getPixelForValue(chart.data.labels[period.start]);
          const endX = scales.x.getPixelForValue(chart.data.labels[period.end]);
          
          // Draw a semi-transparent rectangle for the drawdown period
          ctx.fillStyle = dataset.backgroundColor.replace('0.1', '0.2');
          ctx.fillRect(startX, chartArea.top, endX - startX, chartArea.height);
          
          // Mark the peak with a vertical line
          const peakX = scales.x.getPixelForValue(chart.data.labels[period.peakIndex]);
          ctx.strokeStyle = dataset.borderColor;
          ctx.setLineDash([5, 3]);
          ctx.beginPath();
          ctx.moveTo(peakX, chartArea.top);
          ctx.lineTo(peakX, chartArea.bottom);
          ctx.stroke();
          ctx.setLineDash([]);
        });
      }
    }
  };
  
  // Create returns chart with drawdown periods highlighted
  const mddReturnsChart = new Chart(mddReturnsChartCtx, {
    type: 'line',
    plugins: [drawdownPeriodPlugin],
    data: {
      labels: syntheticData.dates,
      datasets: [
        {
          label: 'Asset D',
          data: syntheticData.cumulative.assetD,
          borderColor: '#ff6384',
          backgroundColor: 'rgba(255, 99, 132, 0.1)',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.1,
          drawdownPeriods: syntheticData.periods.assetD
        },
        {
          label: 'Asset E',
          data: syntheticData.cumulative.assetE,
          borderColor: '#36a2eb',
          backgroundColor: 'rgba(54, 162, 235, 0.1)',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.1,
          drawdownPeriods: syntheticData.periods.assetE
        },
        {
          label: 'Asset F',
          data: syntheticData.cumulative.assetF,
          borderColor: '#ffcd56',
          backgroundColor: 'rgba(255, 205, 86, 0.1)',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.1,
          drawdownPeriods: syntheticData.periods.assetF
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        tooltip: {
          mode: 'index',
          intersect: false,
          callbacks: {
            afterTitle: function(context) {
              const datasetIndex = context[0].datasetIndex;
              const dataIndex = context[0].dataIndex;
              const value = context[0].dataset.data[dataIndex];
              const maxValue = Math.max(...context[0].dataset.data.slice(0, dataIndex + 1));
              const drawdown = ((maxValue - value) / maxValue * 100).toFixed(2);
              return `Current Drawdown: ${drawdown}%`;
            }
          }
        },
        legend: {
          position: 'top'
        }
      },
      scales: {
        x: {
          type: 'time',
          time: {
            unit: 'week',
            displayFormats: {
              week: 'MMM d'
            },
            tooltipFormat: 'MMM d, yyyy'
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
          suggestedMin: 0.7,
          suggestedMax: 1.5
        }
      }
    }
  });
  
  // Create drawdown chart
  const drawdownChart = new Chart(drawdownChartCtx, {
    type: 'line',
    data: {
      labels: syntheticData.dates,
      datasets: [
        {
          label: `Asset D (Max: ${(syntheticData.maxDrawdowns.assetD * 100).toFixed(1)}%)`,
          data: syntheticData.drawdowns.assetD.map(d => d * 100), // Convert to percentage
          borderColor: '#ff6384',
          backgroundColor: 'rgba(255, 99, 132, 0.1)',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.1,
          fill: 'origin'
        },
        {
          label: `Asset E (Max: ${(syntheticData.maxDrawdowns.assetE * 100).toFixed(1)}%)`,
          data: syntheticData.drawdowns.assetE.map(d => d * 100), // Convert to percentage
          borderColor: '#36a2eb',
          backgroundColor: 'rgba(54, 162, 235, 0.1)',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.1,
          fill: 'origin'
        },
        {
          label: `Asset F (Max: ${(syntheticData.maxDrawdowns.assetF * 100).toFixed(1)}%)`,
          data: syntheticData.drawdowns.assetF.map(d => d * 100), // Convert to percentage
          borderColor: '#ffcd56',
          backgroundColor: 'rgba(255, 205, 86, 0.1)',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.1,
          fill: 'origin'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        tooltip: {
          mode: 'index',
          intersect: false,
          callbacks: {
            label: function(context) {
              return `${context.dataset.label.split(' (')[0]}: ${context.parsed.y.toFixed(2)}%`;
            }
          }
        },
        legend: {
          position: 'top'
        }
      },
      scales: {
        x: {
          type: 'time',
          time: {
            unit: 'week',
            displayFormats: {
              week: 'MMM d'
            },
            tooltipFormat: 'MMM d, yyyy'
          },
          title: {
            display: true,
            text: 'Date'
          }
        },
        y: {
          title: {
            display: true,
            text: 'Drawdown (%)'
          },
          min: 0,
          max: 35, // Max drawdown percentage to show
          reverse: true // Invert axis so drawdowns go downward
        }
      }
    }
  });
}

// Utility function to calculate turnover ratio
function calculateTurnover(currentWeights, previousWeights) {
  if (!previousWeights) {
    return 0; // First period has no turnover
  }
  
  let sumAbsDiff = 0;
  for (let i = 0; i < currentWeights.length; i++) {
    sumAbsDiff += Math.abs(currentWeights[i] - previousWeights[i]);
  }
  
  // Multiply by 0.5 to get the turnover ratio (representing minimum fraction of portfolio that must be traded)
  return sumAbsDiff * 0.5;
}

// Generate data for turnover ratio demonstration
function generateTurnoverDemoData(days = 60) {
  const dates = [];
  const today = new Date();
  
  // Generate dates
  for (let i = days; i >= 0; i--) {
    const date = new Date(today);
    date.setDate(today.getDate() - i);
    dates.push(date);
  }
  
  // Portfolio 1: Stable allocations around 40%, 35%, 25%
  const portfolio1Weights = [];
  const portfolio1Turnover = [];
  let prevWeights1 = null;
  
  for (let i = 0; i <= days; i++) {
    // Target weights with small random fluctuations
    const target1 = 0.40;
    const target2 = 0.35;
    const target3 = 0.25;
    
    // Small random adjustments (up to +/- 2%)
    const adjust1 = (Math.random() - 0.5) * 0.02;
    const adjust2 = (Math.random() - 0.5) * 0.02;
    
    // Calculate weights ensuring they sum to 1
    let weight1 = Math.max(0.01, Math.min(0.99, target1 + adjust1));
    let weight2 = Math.max(0.01, Math.min(0.99 - weight1, target2 + adjust2));
    let weight3 = 1 - weight1 - weight2;
    
    const currentWeights = [weight1, weight2, weight3];
    portfolio1Weights.push(currentWeights);
    
    // Calculate turnover
    const turnover = calculateTurnover(currentWeights, prevWeights1);
    portfolio1Turnover.push(turnover);
    
    prevWeights1 = currentWeights;
  }
  
  // Portfolio 2: Erratic trading with large daily changes
  const portfolio2Weights = [];
  const portfolio2Turnover = [];
  let prevWeights2 = null;
  
  for (let i = 0; i <= days; i++) {
    // Significantly different weights each day
    let weight1 = Math.random() * 0.8 + 0.1; // Between 10% and 90%
    let weight2 = Math.random() * (1 - weight1 - 0.05); // Remaining, leaving at least 5% for asset 3
    let weight3 = 1 - weight1 - weight2;
    
    const currentWeights = [weight1, weight2, weight3];
    portfolio2Weights.push(currentWeights);
    
    // Calculate turnover
    const turnover = calculateTurnover(currentWeights, prevWeights2);
    portfolio2Turnover.push(turnover);
    
    prevWeights2 = currentWeights;
  }
  
  // Portfolio 3: Gradual concentration from equal weights to all in one asset
  const portfolio3Weights = [];
  const portfolio3Turnover = [];
  let prevWeights3 = null;
  
  for (let i = 0; i <= days; i++) {
    // Start with equal weights
    if (i === 0) {
      const currentWeights = [1/3, 1/3, 1/3];
      portfolio3Weights.push(currentWeights);
      portfolio3Turnover.push(0);
      prevWeights3 = currentWeights;
      continue;
    }
    
    // Gradually increase weight of asset 1, reduce others
    const progressFactor = i / days; // 0 at start, 1 at end
    let weight1 = 1/3 + progressFactor * 2/3; // Grows from 1/3 to 1
    let weight2 = 1/3 * (1 - progressFactor); // Shrinks from 1/3 to 0
    let weight3 = 1 - weight1 - weight2;      // Shrinks from 1/3 to 0
    
    const currentWeights = [weight1, weight2, weight3];
    portfolio3Weights.push(currentWeights);
    
    // Calculate turnover
    const turnover = calculateTurnover(currentWeights, prevWeights3);
    portfolio3Turnover.push(turnover);
    
    prevWeights3 = currentWeights;
  }
  
  return {
    dates,
    portfolios: {
      stable: {
        weights: portfolio1Weights,
        turnover: portfolio1Turnover
      },
      erratic: {
        weights: portfolio2Weights,
        turnover: portfolio2Turnover
      },
      concentration: {
        weights: portfolio3Weights,
        turnover: portfolio3Turnover
      }
    }
  };
}

// Initialize the Turnover Ratio demonstration charts
function initializeTurnoverDemo() {
  console.log('Initializing Turnover demo');
  const weightsChart1Ctx = document.getElementById('weightsChart1');
  const turnoverChart1Ctx = document.getElementById('turnoverChart1');
  const weightsChart2Ctx = document.getElementById('weightsChart2');
  const turnoverChart2Ctx = document.getElementById('turnoverChart2');
  const weightsChart3Ctx = document.getElementById('weightsChart3');
  const turnoverChart3Ctx = document.getElementById('turnoverChart3');
  
  if (!weightsChart1Ctx || !turnoverChart1Ctx || !weightsChart2Ctx || 
      !turnoverChart2Ctx || !weightsChart3Ctx || !turnoverChart3Ctx) {
    console.error('Chart canvas elements not found for Turnover demo');
    return;
  }
  
  // Generate data for all portfolios
  const data = generateTurnoverDemoData();
  
  // Define colors for assets
  const assetColors = [
    '#8844ee',
    '#44bbee',
    '#ee7744'
  ];
  
  // Create stacked area charts for portfolio weights
  const createWeightsChart = (ctx, weights, title) => {
    const datasets = [];
    
    // Create a dataset for each asset
    for (let assetIndex = 0; assetIndex < 3; assetIndex++) {
      datasets.push({
        label: `Asset ${assetIndex + 1}`,
        data: weights.map(w => w[assetIndex] * 100), // Convert to percentage
        backgroundColor: assetColors[assetIndex],
        borderColor: assetColors[assetIndex],
        borderWidth: 1,
        fill: true,
        tension: 0
      });
    }
    
    return new Chart(ctx, {
      type: 'line',
      data: {
        labels: data.dates,
        datasets: datasets
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            type: 'time',
            time: {
              unit: 'day',
              displayFormats: {
                day: 'MMM d'
              },
              tooltipFormat: 'MMM d, yyyy'
            },
            title: {
              display: true,
              text: 'Date'
            },
            stacked: true
          },
          y: {
            stacked: true,
            min: 0,
            max: 100,
            title: {
              display: true,
              text: 'Allocation (%)'
            }
          }
        },
        plugins: {
          tooltip: {
            mode: 'index',
            intersect: false,
            callbacks: {
              label: function(context) {
                return `${context.dataset.label}: ${context.parsed.y.toFixed(1)}%`;
              }
            }
          },
          legend: {
            position: 'top'
          }
        }
      }
    });
  };
  
  // Create line charts for turnover values
  const createTurnoverChart = (ctx, turnover, title) => {
    return new Chart(ctx, {
      type: 'line',
      data: {
        labels: data.dates,
        datasets: [{
          label: 'Turnover Ratio',
          data: turnover.map(t => t * 100), // Convert to percentage
          borderColor: '#3f88e2',
          backgroundColor: 'rgba(63, 136, 226, 0.1)',
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.1,
          fill: 'origin'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            type: 'time',
            time: {
              unit: 'day',
              displayFormats: {
                day: 'MMM d'
              },
              tooltipFormat: 'MMM d, yyyy'
            },
            title: {
              display: true,
              text: 'Date'
            }
          },
          y: {
            min: 0,
            max: 100,
            title: {
              display: true,
              text: 'Turnover (%)'
            }
          }
        },
        plugins: {
          tooltip: {
            mode: 'index',
            intersect: false,
            callbacks: {
              label: function(context) {
                return `Turnover: ${context.parsed.y.toFixed(1)}%`;
              }
            }
          },
          legend: {
            display: false
          }
        }
      }
    });
  };
  
  // Create all charts
  createWeightsChart(weightsChart1Ctx, data.portfolios.stable.weights, 'Stable Portfolio');
  createTurnoverChart(turnoverChart1Ctx, data.portfolios.stable.turnover, 'Stable Portfolio Turnover');
  
  createWeightsChart(weightsChart2Ctx, data.portfolios.erratic.weights, 'Erratic Portfolio');
  createTurnoverChart(turnoverChart2Ctx, data.portfolios.erratic.turnover, 'Erratic Portfolio Turnover');
  
  createWeightsChart(weightsChart3Ctx, data.portfolios.concentration.weights, 'Concentration Portfolio');
  createTurnoverChart(turnoverChart3Ctx, data.portfolios.concentration.turnover, 'Concentration Portfolio Turnover');
}

// Utility function to calculate HHI (Herfindahl-Hirschman Index)
function calculateHHI(weights) {
  return weights.reduce((sum, weight) => sum + weight * weight, 0);
}

// Utility function to calculate ENP (Effective Number of Positions)
function calculateENP(weights, epsilon = 0.0001) {
  const hhi = calculateHHI(weights);
  return 1 / (hhi + epsilon);
}

// Utility function to calculate concentration penalty
function calculateConcentrationPenalty(weights, enpMin, enpMax, epsilon = 0.0001) {
  const enp = calculateENP(weights, epsilon);
  const lowerPenalty = Math.max(enpMin - enp, 0);
  const upperPenalty = Math.max(enp - enpMax, 0);
  return lowerPenalty + upperPenalty;
}

// Initialize the Concentration Penalty demonstration
function initializeConcentrationDemo() {
  console.log('Initializing Concentration demo');
  const concentrationWeightsChartCtx = document.getElementById('concentrationWeightsChart');
  const enpChartCtx = document.getElementById('enpChart');
  const hhiEnpChartCtx = document.getElementById('hhiEnpChart');
  
  if (!concentrationWeightsChartCtx || !enpChartCtx || !hhiEnpChartCtx) {
    console.error('Concentration chart canvas elements not found', {
      concentrationWeightsChartCtx,
      enpChartCtx,
      hhiEnpChartCtx
    });
    return;
  }
  
  // Define ENP target range
  const enpMin = 3;
  const enpMax = 8;
  
  // Define example portfolios with different concentration levels
  const portfolios = [
    {
      name: 'Highly Concentrated',
      weights: [0.65, 0.20, 0.07, 0.03, 0.02, 0.01, 0.01, 0.01],
      color: '#ff6384'
    },
    {
      name: 'Balanced',
      weights: [0.25, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05],
      color: '#4caf50'
    },
    {
      name: 'Overly Diversified',
      weights: [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125],
      color: '#ffcd56'
    }
  ];
  
  // Calculate HHI and ENP for each portfolio
  portfolios.forEach(portfolio => {
    portfolio.hhi = calculateHHI(portfolio.weights);
    portfolio.enp = calculateENP(portfolio.weights);
    portfolio.penalty = calculateConcentrationPenalty(portfolio.weights, enpMin, enpMax);
  });
  
  // Create weight distribution chart
  const concentrationWeightsChart = new Chart(concentrationWeightsChartCtx, {
    type: 'bar',
    data: {
      labels: ['Asset 1', 'Asset 2', 'Asset 3', 'Asset 4', 'Asset 5', 'Asset 6', 'Asset 7', 'Asset 8'],
      datasets: portfolios.map(portfolio => ({
        label: portfolio.name,
        data: portfolio.weights.map(w => w * 100), // Convert to percentage
        backgroundColor: portfolio.color,
        borderColor: portfolio.color,
        borderWidth: 1
      }))
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: {
            display: true,
            text: 'Assets'
          }
        },
        y: {
          title: {
            display: true,
            text: 'Weight (%)'
          },
          min: 0,
          max: 70
        }
      },
      plugins: {
        legend: {
          position: 'top'
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              return `${context.dataset.label}: ${context.parsed.y.toFixed(1)}%`;
            }
          }
        }
      }
    }
  });
  
  // Create ENP comparison chart
  const enpChart = new Chart(enpChartCtx, {
    type: 'bar',
    data: {
      labels: portfolios.map(p => p.name),
      datasets: [
        {
          label: 'Effective Number of Positions',
          data: portfolios.map(p => p.enp),
          backgroundColor: portfolios.map(p => p.color),
          borderColor: portfolios.map(p => p.color),
          borderWidth: 1
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: {
            display: true,
            text: 'Portfolio'
          }
        },
        y: {
          title: {
            display: true,
            text: 'ENP'
          },
          min: 0,
          max: 10,
          ticks: {
            callback: function(value) {
              return value.toFixed(1);
            }
          }
        }
      },
      plugins: {
        annotation: {
          annotations: {
            minLine: {
              type: 'line',
              mode: 'horizontal',
              scaleID: 'y',
              value: enpMin,
              borderColor: '#ff9800',
              borderWidth: 2,
              borderDash: [5, 5],
              label: {
                content: 'ENP Min = ' + enpMin,
                enabled: true,
                position: 'end'
              }
            },
            maxLine: {
              type: 'line',
              mode: 'horizontal',
              scaleID: 'y',
              value: enpMax,
              borderColor: '#8bc34a',
              borderWidth: 2,
              borderDash: [5, 5],
              label: {
                content: 'ENP Max = ' + enpMax,
                enabled: true,
                position: 'end'
              }
            }
          }
        },
        legend: {
          display: false
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              const portfolio = portfolios[context.dataIndex];
              let label = `ENP: ${context.parsed.y.toFixed(2)}`;
              if (portfolio.penalty > 0) {
                label += ` (Penalty: ${portfolio.penalty.toFixed(2)})`;
              }
              return label;
            },
            afterLabel: function(context) {
              const portfolio = portfolios[context.dataIndex];
              return `HHI: ${portfolio.hhi.toFixed(3)}`;
            }
          }
        }
      }
    }
  });
  
  // Create HHI vs ENP relationship chart
  // Generate data points for the inverse relationship curve
  const hhiValues = [];
  const enpValues = [];
  
  // Generate data points from HHI = 0.125 (equal weights) to HHI = 1 (complete concentration)
  for (let hhi = 0.125; hhi <= 1; hhi += 0.01) {
    hhiValues.push(hhi);
    enpValues.push(1 / hhi);
  }
  
  // Add actual portfolio data points
  const portfolioPoints = portfolios.map(p => ({
    x: p.hhi,
    y: p.enp
  }));
  
  const hhiEnpChart = new Chart(hhiEnpChartCtx, {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: 'HHI-ENP Curve',
          data: hhiValues.map((hhi, i) => ({ x: hhi, y: enpValues[i] })),
          showLine: true,
          borderColor: '#3f88e2',
          backgroundColor: 'rgba(63, 136, 226, 0.1)',
          borderWidth: 2,
          pointRadius: 0
        },
        {
          label: 'Portfolio Examples',
          data: portfolioPoints,
          borderColor: '#ffffff',
          backgroundColor: portfolios.map(p => p.color),
          pointRadius: 6,
          pointHoverRadius: 8
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          type: 'linear',
          title: {
            display: true,
            text: 'HHI (Herfindahl-Hirschman Index)'
          },
          min: 0,
          max: 1
        },
        y: {
          title: {
            display: true,
            text: 'ENP (Effective Number of Positions)'
          },
          min: 0,
          max: 10
        }
      },
      plugins: {
        tooltip: {
          callbacks: {
            label: function(context) {
              if (context.datasetIndex === 0) {
                return `HHI: ${context.parsed.x.toFixed(3)}, ENP: ${context.parsed.y.toFixed(2)}`;
              } else {
                const portfolio = portfolios[context.dataIndex];
                return `${portfolio.name}: HHI = ${portfolio.hhi.toFixed(3)}, ENP = ${portfolio.enp.toFixed(2)}`;
              }
            }
          }
        }
      }
    }
  });
} 