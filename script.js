// ============================================
// KNN Classification Dashboard - JavaScript
// ============================================

// Configuration
const API_BASE = '/api';
let modelData = {};

// ============================================
// Initialization
// ============================================

document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Dashboard initialized');
    
    // Initialize model data
    loadModelInfo();
    loadVisualizations();
    setupEventListeners();
});

// ============================================
// API Calls
// ============================================

async function loadModelInfo() {
    try {
        const response = await fetch(`${API_BASE}/model/info`);
        const data = await response.json();
        
        if (data.status === 'success') {
            modelData = data;
            updateStats();
            updateMetrics();
        }
    } catch (error) {
        console.error('Error loading model info:', error);
    }
}

async function loadVisualizations() {
    try {
        // Load decision boundary
        const boundaryResponse = await fetch(`${API_BASE}/plot/decision_boundary`);
        const boundaryData = await boundaryResponse.json();
        
        if (boundaryData.status === 'success') {
            document.getElementById('boundary-plot').src = boundaryData.image;
            document.getElementById('decision-boundary').querySelector('.plot-loader').style.display = 'none';
        }
        
        // Load confusion matrix
        const confusionResponse = await fetch(`${API_BASE}/plot/confusion_matrix`);
        const confusionData = await confusionResponse.json();
        
        if (confusionData.status === 'success') {
            document.getElementById('confusion-plot').src = confusionData.image;
            document.getElementById('confusion-matrix').querySelector('.plot-loader').style.display = 'none';
        }
    } catch (error) {
        console.error('Error loading visualizations:', error);
    }
}

async function makePrediction() {
    try {
        const feature1 = parseFloat(document.getElementById('feature1').value);
        const feature2 = parseFloat(document.getElementById('feature2').value);
        
        const response = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                feature1: feature1,
                feature2: feature2
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            displayPrediction(data.prediction);
        }
    } catch (error) {
        console.error('Error making prediction:', error);
    }
}

// ============================================
// UI Updates
// ============================================

function updateStats() {
    // Format accuracy as percentage
    const trainAccuracyPercent = (modelData.train_accuracy * 100).toFixed(2);
    const testAccuracyPercent = (modelData.test_accuracy * 100).toFixed(2);
    
    document.getElementById('train-acc').textContent = `${trainAccuracyPercent}%`;
    document.getElementById('test-acc').textContent = `${testAccuracyPercent}%`;
    document.getElementById('train-samples').textContent = modelData.train_samples.toLocaleString();
    document.getElementById('test-samples').textContent = modelData.test_samples.toLocaleString();
}

function updateMetrics() {
    // Model configuration
    document.getElementById('k-value').textContent = modelData.k_neighbors;
    document.getElementById('n-features').textContent = modelData.n_features;
    document.getElementById('n-classes').textContent = modelData.n_classes;
    
    // Data split information
    const totalSamples = modelData.train_samples + modelData.test_samples;
    document.getElementById('train-count').textContent = modelData.train_samples.toLocaleString();
    document.getElementById('test-count').textContent = modelData.test_samples.toLocaleString();
    document.getElementById('total-count').textContent = totalSamples.toLocaleString();
    
    // Classification metrics
    if (modelData.classification_report) {
        displayClassificationMetrics(modelData.classification_report);
    }
}

function displayClassificationMetrics(report) {
    const metricsContainer = document.getElementById('classification-metrics');
    
    let html = `
        <table style="width: 100%; font-size: 0.85rem;">
            <tr style="border-bottom: 1px solid rgba(102, 126, 234, 0.2);">
                <td style="padding: 0.5rem 0; color: var(--color-text-muted);">Class</td>
                <td style="padding: 0.5rem 0; text-align: right; color: var(--color-text-muted);">Precision</td>
                <td style="padding: 0.5rem 0; text-align: right; color: var(--color-text-muted);">Recall</td>
                <td style="padding: 0.5rem 0; text-align: right; color: var(--color-text-muted);">F1-Score</td>
            </tr>
    `;
    
    // Add individual class rows
    for (let i = 0; i < modelData.n_classes; i++) {
        const classMetrics = report[i];
        html += `
            <tr style="border-bottom: 1px solid rgba(102, 126, 234, 0.1);">
                <td style="padding: 0.75rem 0; color: var(--color-text);">Class ${i}</td>
                <td style="padding: 0.75rem 0; text-align: right; color: var(--color-accent-2);">
                    ${(classMetrics.precision * 100).toFixed(1)}%
                </td>
                <td style="padding: 0.75rem 0; text-align: right; color: var(--color-accent-2);">
                    ${(classMetrics.recall * 100).toFixed(1)}%
                </td>
                <td style="padding: 0.75rem 0; text-align: right; color: var(--color-accent-2);">
                    ${(classMetrics['f1-score'] * 100).toFixed(1)}%
                </td>
            </tr>
        `;
    }
    
    // Add weighted average
    const weightedAvg = report['weighted avg'];
    html += `
        <tr>
            <td style="padding: 0.75rem 0; color: var(--color-text); font-weight: 700;">Overall</td>
            <td style="padding: 0.75rem 0; text-align: right; color: var(--color-primary); font-weight: 700;">
                ${(weightedAvg.precision * 100).toFixed(1)}%
            </td>
            <td style="padding: 0.75rem 0; text-align: right; color: var(--color-primary); font-weight: 700;">
                ${(weightedAvg.recall * 100).toFixed(1)}%
            </td>
            <td style="padding: 0.75rem 0; text-align: right; color: var(--color-primary); font-weight: 700;">
                ${(weightedAvg['f1-score'] * 100).toFixed(1)}%
            </td>
        </tr>
        </table>
    `;
    
    metricsContainer.innerHTML = html;
}

function displayPrediction(prediction) {
    const resultContainer = document.getElementById('result-container');
    const predictionClass = document.getElementById('prediction-class');
    
    resultContainer.classList.remove('result-hidden');
    resultContainer.classList.add('result-active');
    
    // Color based on prediction
    const color = prediction === 0 ? 'var(--color-accent-1)' : 'var(--color-accent-2)';
    predictionClass.textContent = `Class ${prediction}`;
    predictionClass.style.color = color;
}

// ============================================
// Event Listeners
// ============================================

function setupEventListeners() {
    // Tab switching
    document.querySelectorAll('.tab-button').forEach(button => {
        button.addEventListener('click', (e) => {
            // Remove active class from all tabs and buttons
            document.querySelectorAll('.tab-button').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            
            // Add active class to clicked button and corresponding content
            e.target.classList.add('active');
            const tabId = e.target.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');
        });
    });
    
    // Prediction sliders
    const feature1Input = document.getElementById('feature1');
    const feature2Input = document.getElementById('feature2');
    const feature1Value = document.getElementById('feature1-value');
    const feature2Value = document.getElementById('feature2-value');
    
    feature1Input.addEventListener('input', (e) => {
        feature1Value.textContent = parseFloat(e.target.value).toFixed(1);
    });
    
    feature2Input.addEventListener('input', (e) => {
        feature2Value.textContent = parseFloat(e.target.value).toFixed(1);
    });
    
    // Predict button
    document.getElementById('predict-btn').addEventListener('click', makePrediction);
    
    // Allow Enter key on sliders
    [feature1Input, feature2Input].forEach(input => {
        input.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                makePrediction();
            }
        });
    });
    
    // Smooth scroll for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', (e) => {
            const href = anchor.getAttribute('href');
            if (href !== '#' && document.querySelector(href)) {
                e.preventDefault();
                document.querySelector(href).scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

// ============================================
// Utilities
// ============================================

// Format numbers with thousand separators
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

// Log helper for debugging
function logDebug(message, data = null) {
    console.log(`[KNN Dashboard] ${message}`, data || '');
}

// ============================================
// Error Handling
// ============================================

window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
});

window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
});
