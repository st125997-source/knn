#!/usr/bin/env python3
"""
Flask web application for KNN Classification Pipeline
Serves HTML/CSS/JS dashboard and exposes ML model via API
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import io
import base64
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
import os

# Initialize Flask app
app = Flask(__name__, 
            static_folder='static',
            static_url_path='/static',
            template_folder='templates')

# Set random seed
np.random.seed(42)

# Global variables to store model and data
model = None
X_train, X_test = None, None
y_train, y_test = None, None
train_accuracy, test_accuracy = None, None

def initialize_model():
    """Initialize and train the KNN pipeline"""
    global model, X_train, X_test, y_train, y_test, train_accuracy, test_accuracy
    
    # Generate data
    X, y = make_moons(n_samples=2000, noise=0.3, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train pipeline
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ])
    model.fit(X_train, y_train)
    
    # Calculate accuracies
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    
    print("✓ Model initialized and trained")
    print(f"  Train Accuracy: {train_accuracy:.4f}")
    print(f"  Test Accuracy: {test_accuracy:.4f}")

def generate_plot(plot_type='decision_boundary'):
    """Generate plots and return as base64"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if plot_type == 'decision_boundary':
        # Generate mesh
        h = 0.02
        x_min, x_max = X_test[:, 0].min() - 0.5, X_test[:, 0].max() + 0.5
        y_min, y_max = X_test[:, 1].min() - 0.5, X_test[:, 1].max() + 0.5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        # Get predictions
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # Plot
        ax.contourf(xx, yy, Z, alpha=0.3, levels=1, colors=['#FF6B6B', '#4ECDC4'])
        
        # Scatter plot
        scatter1 = ax.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], 
                            c='#FF6B6B', s=100, alpha=0.7, edgecolors='black', linewidth=1.5,
                            label='Class 0')
        scatter2 = ax.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], 
                            c='#4ECDC4', s=100, alpha=0.7, edgecolors='black', linewidth=1.5,
                            label='Class 1')
        
        ax.set_title('KNN Decision Boundary (Test Set)', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Feature 1', fontsize=11)
        ax.set_ylabel('Feature 2', fontsize=11)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(alpha=0.2)
        
    elif plot_type == 'confusion_matrix':
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        im = ax.imshow(cm, interpolation='nearest', cmap='RdYlGn', aspect='auto')
        ax.set_xlabel('Predicted Label', fontsize=11)
        ax.set_ylabel('True Label', fontsize=11)
        ax.set_title('Confusion Matrix (Test Set)', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, cm[i, j],
                             ha="center", va="center", color="black", fontsize=12, fontweight='bold')
        
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    # Convert to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    
    return f"data:image/png;base64,{image_base64}"

# ==================== ROUTES ====================

@app.route('/')
def index():
    """Serve the main dashboard"""
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

@app.route('/api/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    y_pred = model.predict(X_test)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    return jsonify({
        'status': 'success',
        'train_accuracy': float(train_accuracy),
        'test_accuracy': float(test_accuracy),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'n_features': X_train.shape[1],
        'n_classes': len(np.unique(y_test)),
        'k_neighbors': 5,
        'classification_report': class_report
    })

@app.route('/api/plot/<plot_type>', methods=['GET'])
def get_plot(plot_type):
    """Get generated plot"""
    if plot_type not in ['decision_boundary', 'confusion_matrix']:
        return jsonify({'error': 'Invalid plot type'}), 400
    
    plot_data = generate_plot(plot_type)
    return jsonify({
        'status': 'success',
        'image': plot_data
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction on new data"""
    try:
        data = request.json
        features = np.array([[data['feature1'], data['feature2']]])
        
        # Scale using the model's scaler
        features_scaled = model.named_steps['scaler'].transform(features)
        prediction = model.named_steps['knn'].predict(features_scaled)[0]
        
        return jsonify({
            'status': 'success',
            'prediction': int(prediction),
            'class_name': f'Class {int(prediction)}'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/data/summary', methods=['GET'])
def data_summary():
    """Get data summary statistics"""
    return jsonify({
        'status': 'success',
        'X_train_shape': list(X_train.shape),
        'X_test_shape': list(X_test.shape),
        'feature_1_min': float(np.min(X_test[:, 0])),
        'feature_1_max': float(np.max(X_test[:, 0])),
        'feature_2_min': float(np.min(X_test[:, 1])),
        'feature_2_max': float(np.max(X_test[:, 1]))
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Server error'}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("KNN Classification Pipeline - Web Dashboard")
    print("=" * 60)
    initialize_model()
    print("\n🚀 Starting Flask server...")
    print("📊 Dashboard: http://localhost:5000")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False)
