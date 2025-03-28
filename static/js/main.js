/////////////////////////////////////////////////////////////////////////////////////////////////////////
// File Name - main.js
// Author - Ojas Ulhas Dighe
// Date - 3rd Mar 2025
// Description - This file contains the main logic for the frontend of the application.
/////////////////////////////////////////////////////////////////////////////////////////////////////////


/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Fucntion Name - formatCurrency
// Author - Ojas Ulhas Dighe
// Date - 3rd Mar 2025
// Description - This function is used to format the currency in the form of ₹.
/////////////////////////////////////////////////////////////////////////////////////////////////////////

function formatCurrency(value) {
    return `₹${parseFloat(value).toFixed(2)}`;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Name - getRecommendationClass
// Author - Ojas Ulhas Dighe
// Date - 3rd Mar 2025
// Description - This function is used to get the recommendation class based on the recommendation.
/////////////////////////////////////////////////////////////////////////////////////////////////////////

function getRecommendationClass(recommendation) {
    const classes = {
        'Strong Buy': 'strong-buy',
        'Buy': 'buy',
        'Hold': 'hold',
        'Sell': 'sell',
        'Strong Sell': 'strong-sell'
    };
    return classes[recommendation] || '';
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Name - createPriceChart
// Author - Ojas Ulhas Dighe
// Date - 3rd Mar 2025
// Description - This function is used to create the price chart.
/////////////////////////////////////////////////////////////////////////////////////////////////////////

function createPriceChart(chartData) {
    const dates = chartData.map(d => d.date);
    const prices = chartData.map(d => d.price);
    const sma20 = chartData.map(d => d.sma20);
    const sma50 = chartData.map(d => d.sma50);

    const traces = [
        {
            name: 'Price',
            x: dates,
            y: prices,
            type: 'scatter',
            line: { color: '#2563eb' }
        },
        {
            name: '20 SMA',
            x: dates,
            y: sma20,
            type: 'scatter',
            line: { color: '#16a34a' }
        },
        {
            name: '50 SMA',
            x: dates,
            y: sma50,
            type: 'scatter',
            line: { color: '#dc2626' }
        }
    ];

    const layout = {
        title: 'Price Action',
        xaxis: { title: 'Date' },
        yaxis: { title: 'Price (₹)' },
        showlegend: true,
        legend: { orientation: 'h', y: -0.2 }
    };

    Plotly.newPlot('priceChart', traces, layout);
}  

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Name - createIndicatorsChart
// Author - Ojas Ulhas Dighe
// Date - 3rd Mar 2025
// Description - This function is used to create the indicators chart.
/////////////////////////////////////////////////////////////////////////////////////////////////////////

function createIndicatorsChart(chartData) {
    const dates = chartData.map(d => d.date);
    const rsi = chartData.map(d => d.rsi);
    const macd = chartData.map(d => d.macd);
    const signal = chartData.map(d => d.signal);

    const traces = [
        {
            name: 'RSI',
            x: dates,
            y: rsi,
            type: 'scatter',
            yaxis: 'y1',
            line: { color: '#8b5cf6' }
        },
        {
            name: 'MACD',
            x: dates,
            y: macd,
            type: 'scatter',
            yaxis: 'y2',
            line: { color: '#3b82f6' }
        },
        {
            name: 'Signal',
            x: dates,
            y: signal,
            type: 'scatter',
            yaxis: 'y2',
            line: { color: '#ef4444' }
        }
    ];

    const layout = {
        title: 'Technical Indicators',
        xaxis: { title: 'Date' },
        yaxis: { 
            title: 'RSI',
            domain: [0.6, 1]
        },
        yaxis2: {
            title: 'MACD',
            domain: [0, 0.4]
        },
        showlegend: true,
        legend: { orientation: 'h', y: -0.2 },
        grid: { rows: 2, columns: 1, pattern: 'independent' }
    };

    Plotly.newPlot('indicatorsChart', traces, layout);
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Name - analyzeStock
// Author - Ojas Ulhas Dighe
// Date - 3rd Mar 2025
// Description - This function is used to analyze the stock.
/////////////////////////////////////////////////////////////////////////////////////////////////////////

async function analyzeStock() {
    const stockSymbol = document.getElementById('stockSymbol').value;
    const exchange = document.getElementById('exchange').value;
    const startDate = document.getElementById('startDate').value;

    if (!stockSymbol) {
        showError('Please enter a stock symbol');
        return;
    }

    // Show loading state
    document.getElementById('loadingIndicator').classList.remove('hidden');
    document.getElementById('results').classList.add('hidden');
    document.getElementById('errorMessage').classList.add('hidden');

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                symbol: stockSymbol,
                exchange: exchange,
                startDate: startDate
            })
        });

        const result = await response.json();
        console.log(result)
        if (!result.success) {
            throw new Error(result.error);
        }

        updateUI(result.data);
    } catch (error) {
        showError(error.message || 'An error occurred while analyzing the stock');
    } finally {
        document.getElementById('loadingIndicator').classList.add('hidden');
    }
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Name - updateUI
// Author - Ojas Ulhas Dighe
// Date - 3rd Mar 2025
// Description - This function is used to update the UI.
/////////////////////////////////////////////////////////////////////////////////////////////////////////

function updateUI(data) {
    // Show results container
    document.getElementById('results').classList.remove('hidden');
    // show charts
    document.getElementById('charts').style.display = 'grid'

    // Update current price and recommendation
    document.getElementById('currentPrice').textContent = formatCurrency(data.currentPrice);
    const recommendationBadge = document.getElementById('recommendationBadge');
    recommendationBadge.textContent = data.recommendation;
    recommendationBadge.className = `recommendation-badge ${getRecommendationClass(data.recommendation)}`;

    // Update signals list
    const signalsList = document.getElementById('signalsList');
    signalsList.innerHTML = '';
    data.signals.forEach(signal => {
        const li = document.createElement('p');
        li.textContent = signal;
        signalsList.appendChild(li);
    });

    // Update risk metrics
    const metrics = data.riskMetrics;
    document.getElementById('sharpeRatio').textContent = metrics.sharpeRatio.toFixed(2);
    document.getElementById('volatility').textContent = metrics.volatility.toFixed(2) + '%';
    document.getElementById('maxDrawdown').textContent = metrics.maxDrawdown.toFixed(2) + '%';
    document.getElementById('beta').textContent = metrics.beta ? metrics.beta.toFixed(2) : '-';

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Name - createPriceChart
// Author - Ojas Ulhas Dighe
// Date - 3rd Mar 2025
// Description - This function is used to create the price chart.
/////////////////////////////////////////////////////////////////////////////////////////////////////////

    createPriceChart(data.chartData);
    createIndicatorsChart(data.chartData);

    // update new stock recomendation
    // Function to populate the object

    const div = document.getElementById('otherStockRecomendationList');
    
    // Clear existing list items
    div.innerHTML = '';
    
    // Create and append new list items
    for (const [stock, score] of Object.entries(data.topGainerScores)) {
        const li = document.createElement('h4');
        const divElement = document.createElement('div');
        divElement.appendChild(li)
        li.textContent = `${stock}`;
        div.appendChild(divElement);
        li.setAttribute('class', 'list-group-item')
        divElement.setAttribute('class', 'list-group-item-div');
    }

}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
// Function Name - showError
// Author - Ojas Ulhas Dighe
// Date - 3rd Mar 2025
// Description - This function is used to show the error message.
/////////////////////////////////////////////////////////////////////////////////////////////////////////

function showError(message) {
    const errorElement = document.getElementById('errorMessage');
    errorElement.textContent = message;
    errorElement.classList.remove('hidden');
}

// Event Listeners
document.addEventListener('DOMContentLoaded', () => {
    // Set default date to one year ago
    const defaultDate = new Date();
    defaultDate.setFullYear(defaultDate.getFullYear() - 1);
    document.getElementById('startDate').value = defaultDate.toISOString().split('T')[0];

    // Add enter key listener for stock symbol input
    document.getElementById('stockSymbol').addEventListener('keypress', (event) => {
        if (event.key === 'Enter') {
            analyzeStock();
        }
    });
});

// Add window resize handler for charts
let resizeTimeout;
window.addEventListener('resize', () => {
    clearTimeout(resizeTimeout);
    resizeTimeout = setTimeout(() => {
        const results = document.getElementById('results');
        if (!results.classList.contains('hidden')) {
            Plotly.relayout('priceChart', {
                'xaxis.autorange': true,
                'yaxis.autorange': true
            });
            Plotly.relayout('indicatorsChart', {
                'xaxis.autorange': true,
                'yaxis.autorange': true,
                'yaxis2.autorange': true
            });
        }
    }, 250);
});