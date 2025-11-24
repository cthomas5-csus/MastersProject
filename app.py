#!/usr/bin/env python3
"""
Legal Contract to Smart Contract Pipeline
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Add src directory to Python path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename
from config import config

# Import core components
from src.pipeline.integrated_pipeline import IntegratedPipeline, PipelineConfig

def setup_logging(log_level='INFO', log_file='contract_analysis.log'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def create_directories():
    """Create necessary directories"""
    directories = ['uploads', 'pipeline_outputs', 'generated_contracts', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

# Create Flask app
app = Flask(__name__)

# Get configuration
config_name = os.environ.get('FLASK_ENV', 'development')
app_config = config.get(config_name, config['default'])

# Setup logging
setup_logging(app_config.LOG_LEVEL, app_config.LOG_FILE)
logger = logging.getLogger(__name__)

app.config.from_object(app_config)
app.secret_key = app_config.SECRET_KEY

# Create necessary directories
create_directories()

# Initialize pipeline
try:
    pipeline_config = PipelineConfig(
        claude_api_key=app_config.CLAUDE_API_KEY,
        generate_interpretations=bool(app_config.CLAUDE_API_KEY),
        deploy_to_blockchain=False,  # Keep simple for now
        save_outputs=True,
        output_directory='pipeline_outputs',
        enable_bert_ambiguity=True  # Enable BERT ambiguity detection
    )
    
    pipeline = IntegratedPipeline(pipeline_config)
    logger.info("Pipeline initialized successfully")
    
    if app_config.CLAUDE_API_KEY:
        logger.info("Claude API enabled")
    else:
        logger.warning("Claude API disabled - no API key found")
        
except Exception as e:
    logger.error(f"Failed to initialize pipeline: {e}")
    pipeline = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'txt', 'docx', 'pdf'}

# HTML Template
MAIN_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Legal Contract Analysis Pipeline</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        .upload-area {
            border: 2px dashed #ccc;
            padding: 40px;
            text-align: center;
            margin: 20px 0;
            border-radius: 10px;
            transition: border-color 0.3s ease;
        }
        .upload-area:hover {
            border-color: #007bff;
        }
        .status-badge {
            font-size: 0.8em;
        }
    </style>
</head>
<body>
    <div class="container mt-5">
        <div class="row justify-content-center">
            <div class="col-md-10">
                <div class="card">
                    <div class="card-header bg-primary text-white">
                        <h2 class="text-center mb-0">📄 Legal Contract Analysis Pipeline</h2>
                        <p class="text-center mb-0 mt-2">Convert legal contracts to smart contracts with ambiguity detection</p>
                    </div>
                    <div class="card-body">
                        <!-- Status -->
                        <div class="row mb-4">
                            <div class="col-md-6">
                                <span class="badge bg-{{ 'success' if claude_enabled else 'warning' }} status-badge">
                                    🤖 Claude AI: {{ 'Enabled' if claude_enabled else 'Disabled' }}
                                </span>
                            </div>
                            <div class="col-md-6 text-end">
                                <span class="badge bg-{{ 'success' if pipeline_ready else 'danger' }} status-badge">
                                    ⚙️ Pipeline: {{ 'Ready' if pipeline_ready else 'Error' }}
                                </span>
                            </div>
                        </div>
                        
                        <!-- Upload Form -->
                        <form id="uploadForm" enctype="multipart/form-data">
                            <div class="upload-area">
                                <input type="file" class="form-control" id="contractFile" name="file" 
                                       accept=".pdf,.docx,.txt" required style="display: none;">
                                <div id="uploadText">
                                    <h5>Click to select a contract file</h5>
                                    <p class="text-muted">Supported formats: PDF, DOCX, TXT (max 16MB)</p>
                                    <button type="button" class="btn btn-outline-primary" onclick="document.getElementById('contractFile').click();">
                                        📁 Choose File
                                    </button>
                                </div>
                                <div id="fileSelected" style="display: none;">
                                    <h6 class="text-success">✅ File selected: <span id="fileName"></span></h6>

                                    <button type="submit" class="btn btn-success btn-lg mt-2">
                                        🔍 Analyze Contract
                                    </button>
                                </div>
                            </div>
                        </form>
                        
                        <!-- Progress -->
                        <div id="progress" class="mt-3" style="display: none;">
                            <div class="progress">
                                <div class="progress-bar progress-bar-striped progress-bar-animated" 
                                     role="progressbar" style="width: 100%">
                                    Processing contract...
                                </div>
                            </div>
                        </div>
                        
                        <!-- Results -->
                        <div id="results" class="mt-4" style="display: none;"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        // File selection handler
        document.getElementById('contractFile').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                document.getElementById('fileName').textContent = file.name;
                document.getElementById('uploadText').style.display = 'none';
                document.getElementById('fileSelected').style.display = 'block';
            }
        });
        
        // Form submission handler
        document.getElementById('uploadForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const fileInput = document.getElementById('contractFile');
            const file = fileInput.files[0];
            
            if (!file) {
                alert('Please select a file');
                return;
            }
            
            // Show progress
            document.getElementById('progress').style.display = 'block';
            document.getElementById('results').style.display = 'none';
            
            // Create form data
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                // Hide progress
                document.getElementById('progress').style.display = 'none';
                
                // Show results
                displayResults(result);
                
            } catch (error) {
                document.getElementById('progress').style.display = 'none';
                alert('Error: ' + error.message);
            }
        });
        
        function displayResults(result) {
            const resultsDiv = document.getElementById('results');

            if (result.success) {
                const summary = result.analysis_summary || {};
                const smartContract = result.smart_contract_code || 'No smart contract generated';
                const comparisonData = result.comparison_data || {};

                resultsDiv.innerHTML = `
                    <div class="card">
                        <div class="card-header">
                            <h4>📊 Analysis Results</h4>
                        </div>
                        <div class="card-body">
                            <div class="row mb-3">
                                <div class="col-md-3">
                                    <div class="text-center">
                                        <h5 class="text-primary">${summary.total_clauses || 0}</h5>
                                        <small class="text-muted">Total Clauses</small>
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="text-center">
                                        <h5 class="text-warning">${summary.transactional_clauses || 0}</h5>
                                        <small class="text-muted">Transactional</small>
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="text-center">
                                        <h5 class="text-success">${summary.interpretations_generated || 0}</h5>
                                        <small class="text-muted">AI Interpretations</small>
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="text-center">
                                        <h5 class="text-info">${result.processing_time ? result.processing_time.toFixed(2) : '0.00'}s</h5>
                                        <small class="text-muted">Processing Time</small>
                                    </div>
                                </div>
                            </div>

                            ${generateComparisonSection(comparisonData)}

                            <div class="mt-4">
                                <h5>⚡ Generated Smart Contract:</h5>
                                <pre class="bg-light p-3 rounded" style="max-height: 400px; overflow-y: auto;"><code>${smartContract}</code></pre>
                            </div>

                            ${generateBERTAnalysisSection(result.bert_analysis)}
                        </div>
                    </div>
                `;
            } else {
                resultsDiv.innerHTML = `
                    <div class="alert alert-danger">
                        <h5>❌ Analysis Failed</h5>
                        <p>${result.error}</p>
                    </div>
                `;
            }
            
            resultsDiv.style.display = 'block';
        }

        function generateComparisonSection(comparisonData) {
            if (!comparisonData || !comparisonData.clause_comparisons || comparisonData.clause_comparisons.length === 0) {
                return '<div class="mt-4"><h5>🔍 Ambiguity Analysis</h5><div class="alert alert-info">No Claude interpretations available for comparison. Enable Claude API for detailed ambiguity analysis.</div></div>';
            }

            const comparisons = comparisonData.clause_comparisons;
            const improvementStats = comparisonData.improvement_stats || {};
            const adequacy = comparisonData.adequacy || 'incomplete';

            // Generate overview stats
            const overviewHtml = `
                <div class="mt-4">
                    <h5>🔍 Ambiguity Analysis - Original vs Interpreted Clauses</h5>
                    <div class="row mb-3">
                        <div class="col-md-3">
                            <div class="text-center">
                                <h6 class="text-success">${improvementStats.successful_interpretations || 0}</h6>
                                <small class="text-muted">Successful Interpretations</small>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="text-center">
                                <h6 class="text-primary">${improvementStats.clauses_improved || 0}</h6>
                                <small class="text-muted">Clauses Improved</small>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="text-center">
                                <h6 class="text-info">${improvementStats.average_improvement ? improvementStats.average_improvement.toFixed(1) : '0.0'}%</h6>
                                <small class="text-muted">Avg Improvement</small>
                            </div>
                        </div>
                        <div class="col-md-3">
                            <div class="text-center">
                                <span class="badge bg-${getAdequacyColor(adequacy)} p-2">${adequacy.toUpperCase()}</span>
                                <br><small class="text-muted">Adequacy Rating</small>
                            </div>
                        </div>
                    </div>
                </div>
            `;

            // Generate detailed comparisons
            let detailsHtml = `
                <div class="accordion" id="comparisonAccordion">
            `;

            comparisons.forEach((comparison, index) => {
                if (comparison.success && comparison.metrics_comparison) {
                    const metricsComp = comparison.metrics_comparison;
                    const overallImprovement = comparison.overall_improvement || 0;

                    detailsHtml += `
                        <div class="accordion-item">
                            <h2 class="accordion-header" id="heading${index}">
                                <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse"
                                        data-bs-target="#collapse${index}" aria-expanded="false" aria-controls="collapse${index}">
                                    <div class="d-flex w-100 justify-content-between align-items-center">
                                        <span>Clause ${comparison.clause_id} - ${overallImprovement > 0 ? 'Improved' : overallImprovement < 0 ? 'Degraded' : 'Unchanged'}</span>
                                        <span class="badge bg-${overallImprovement > 5 ? 'success' : overallImprovement > 0 ? 'primary' : overallImprovement < -5 ? 'danger' : 'secondary'} ms-2">
                                            ${overallImprovement.toFixed(1)}%
                                        </span>
                                    </div>
                                </button>
                            </h2>
                            <div id="collapse${index}" class="accordion-collapse collapse" aria-labelledby="heading${index}"
                                 data-bs-parent="#comparisonAccordion">
                                <div class="accordion-body">
                                    <div class="row">
                                        <div class="col-md-6">
                                            <h6>📄 Original Clause</h6>
                                            <div class="bg-light p-3 rounded mb-3" style="max-height: 200px; overflow-y: auto;">
                                                ${comparison.original_text || 'N/A'}
                                            </div>
                                        </div>
                                        <div class="col-md-6">
                                            <h6>✨ Interpreted Clause</h6>
                                            <div class="bg-light p-3 rounded mb-3" style="max-height: 200px; overflow-y: auto;">
                                                ${comparison.interpretation_text || 'N/A'}
                                            </div>
                                        </div>
                                    </div>

                                    <h6>📊 Metrics Comparison</h6>
                                    <div class="table-responsive">
                                        <table class="table table-sm">
                                            <thead>
                                                <tr>
                                                    <th>Metric</th>
                                                    <th>Original</th>
                                                    <th>Interpreted</th>
                                                    <th>Change</th>
                                                    <th>Improvement</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                ${generateMetricRows(metricsComp)}
                                            </tbody>
                                        </table>
                                    </div>

                                    ${comparison.ambiguity_reduction ? `
                                        <h6>🎯 Ambiguity Reduction</h6>
                                        <div class="row mb-2">
                                            <div class="col-md-6">
                                                <small>Vague terms reduced: <strong>${comparison.ambiguity_reduction.reduction_count || 0}</strong></small>
                                            </div>
                                            <div class="col-md-6">
                                                <small>Reduction rate: <strong>${(comparison.ambiguity_reduction.reduction_percentage || 0).toFixed(1)}%</strong></small>
                                            </div>
                                        </div>
                                    ` : ''}

                                    ${comparison.interpretation_quality ? `
                                        <h6>⭐ Interpretation Quality</h6>
                                        <div class="row">
                                            <div class="col-md-4">
                                                <small>Quality Score: <span class="badge bg-${getQualityColor(comparison.interpretation_quality.overall_quality_score || 0)}">${(comparison.interpretation_quality.overall_quality_score || 0).toFixed(2)}</span></small>
                                            </div>
                                            <div class="col-md-4">
                                                <small>Content Preservation: <strong>${((comparison.interpretation_quality.content_preservation || 0) * 100).toFixed(1)}%</strong></small>
                                            </div>
                                            <div class="col-md-4">
                                                <small>Rating: <span class="badge bg-secondary">${(comparison.interpretation_quality.quality_rating || 'N/A').toUpperCase()}</span></small>
                                            </div>
                                        </div>
                                    ` : ''}
                                </div>
                            </div>
                        </div>
                    `;
                }
            });

            detailsHtml += `</div>`;

            return overviewHtml + detailsHtml;
        }

        function generateMetricRows(metricsComp) {
            const metrics = [
                { key: 'fog_index', name: 'Readability (FOG)' },
                { key: 'civility', name: 'Civility' },
                { key: 'contractual_clarity', name: 'Clarity' },
                { key: 'contractual_accuracy', name: 'Accuracy' },
                { key: 'risk_factor_score', name: 'Risk Score' },
                { key: 'aggregated_score', name: 'Overall Score' }
            ];

            return metrics.map(metric => {
                const data = metricsComp[metric.key];
                if (!data) return '';

                const change = data.absolute_change || 0;
                const improvement = data.percent_change || 0;

                return `
                    <tr>
                        <td><strong>${metric.name}</strong></td>
                        <td>${data.original ? data.original.toFixed(3) : '0.000'}</td>
                        <td>${data.interpreted ? data.interpreted.toFixed(3) : '0.000'}</td>
                        <td class="${change > 0 ? 'text-success' : change < 0 ? 'text-danger' : 'text-muted'}">
                            ${change > 0 ? '+' : ''}${change.toFixed(3)}
                        </td>
                        <td class="${improvement > 0 ? 'text-success' : improvement < 0 ? 'text-danger' : 'text-muted'}">
                            ${improvement > 0 ? '+' : ''}${improvement.toFixed(1)}%
                        </td>
                    </tr>
                `;
            }).join('');
        }

        function generateBERTAnalysisSection(bertAnalysis) {
            if (!bertAnalysis || bertAnalysis.error) {
                return `
                    <div class="mt-4">
                        <h5>🧠 BERT Ambiguity Analysis</h5>
                        <div class="alert alert-info">
                            ${bertAnalysis?.error ? `BERT analysis failed: ${bertAnalysis.error}` : 'BERT ambiguity analysis not available'}
                        </div>
                    </div>
                `;
            }

            if (!bertAnalysis.success) {
                return `
                    <div class="mt-4">
                        <h5>🧠 BERT Ambiguity Analysis</h5>
                        <div class="alert alert-warning">
                            BERT analysis could not be completed
                        </div>
                    </div>
                `;
            }

            const summary = bertAnalysis.summary || {};
            const clauseByClause = bertAnalysis.clause_by_clause || [];

            return `
                <div class="mt-4">
                    <h5>🧠 BERT Ambiguity Analysis</h5>

                    <!-- Summary Table -->
                    <div class="mb-3">
                        <table class="table table-striped table-hover">
                            <thead class="table-dark">
                                <tr>
                                    <th scope="col">Metric</th>
                                    <th scope="col" class="text-center">Value</th>
                                    <th scope="col" class="text-center">Status</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td><strong>Clauses Analyzed</strong></td>
                                    <td class="text-center"><span class="h5 text-primary">${bertAnalysis.clauses_analyzed || 0}</span></td>
                                    <td class="text-center"><span class="badge bg-primary">Total</span></td>
                                </tr>
                                <tr>
                                    <td><strong>Ambiguity Reduction</strong></td>
                                    <td class="text-center"><span class="h5 text-${summary.improvement_percentage > 0 ? 'success' : 'warning'}">${(summary.improvement_percentage || 0).toFixed(1)}%</span></td>
                                    <td class="text-center"><span class="badge bg-${summary.improvement_percentage > 0 ? 'success' : 'warning'}">${summary.improvement_percentage > 0 ? 'Improved' : 'No Change'}</span></td>
                                </tr>
                                <tr>
                                    <td><strong>Clauses Improved</strong></td>
                                    <td class="text-center"><span class="h5 text-info">${summary.clauses_improved || 0}</span></td>
                                    <td class="text-center"><span class="badge bg-info">${summary.clauses_improved > 0 ? 'Enhanced' : 'Unchanged'}</span></td>
                                </tr>
                                <tr>
                                    <td><strong>Vague Terms Eliminated</strong></td>
                                    <td class="text-center"><span class="h5 text-success">${summary.vague_terms_eliminated || 0}</span></td>
                                    <td class="text-center"><span class="badge bg-success">${summary.vague_terms_eliminated > 0 ? 'Eliminated' : 'None'}</span></td>
                                </tr>
                            </tbody>
                        </table>
                    </div>

                    <!-- Detailed Analysis -->
                    <div class="row mb-3">
                        <div class="col-md-6">
                            <div class="alert alert-${summary.overall_success ? 'success' : 'warning'}">
                                <strong>Overall Result:</strong>
                                ${summary.overall_success ? '✅ Ambiguity Successfully Reduced' : '⚠️ Limited Improvement Achieved'}
                            </div>
                        </div>
                        <div class="col-md-6">
                            <div class="alert alert-info">
                                <strong>Progress:</strong>
                                ${summary.original_ambiguous_clauses || 0} → ${summary.interpreted_ambiguous_clauses || 0} ambiguous clauses
                            </div>
                        </div>
                    </div>

                    <!-- Clause-by-Clause Dropdowns -->
                    ${clauseByClause.length > 0 ? `
                        <h6>📋 Detailed Clause Analysis</h6>
                        <div class="accordion" id="bertAnalysisAccordion">
                            ${clauseByClause.map((clause, index) => `
                                <div class="accordion-item">
                                    <h2 class="accordion-header" id="bertHeading${index}">
                                        <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse"
                                                data-bs-target="#bertCollapse${index}" aria-expanded="false" aria-controls="bertCollapse${index}">
                                            <div class="d-flex w-100 justify-content-between align-items-center">
                                                <span>Clause ${clause.clause_id} - ${getImprovementStatusText(clause.improvement_status)}</span>
                                                <span class="badge bg-${getImprovementStatusColor(clause.improvement_status)} ms-2">
                                                    ${getImprovementIcon(clause.improvement_status)}
                                                </span>
                                            </div>
                                        </button>
                                    </h2>
                                    <div id="bertCollapse${index}" class="accordion-collapse collapse" aria-labelledby="bertHeading${index}"
                                         data-bs-parent="#bertAnalysisAccordion">
                                        <div class="accordion-body">

                                            <!-- Original vs Interpreted -->
                                            <div class="row mb-3">
                                                <div class="col-md-6">
                                                    <h6>📄 Original Clause</h6>
                                                    <div class="bg-light p-3 rounded mb-2" style="max-height: 150px; overflow-y: auto;">
                                                        ${clause.original_text || 'N/A'}
                                                    </div>
                                                    <div class="mb-2">
                                                        <small class="text-${clause.original_ambiguous ? 'warning' : 'success'}">
                                                            ${clause.original_ambiguous ? '⚠️ Contains ambiguous terms' : '✅ No ambiguous terms'}
                                                        </small>
                                                    </div>
                                                    ${clause.original_ambiguous_terms && clause.original_ambiguous_terms.length > 0 ? `
                                                        <div>
                                                            <small><strong>Ambiguous terms:</strong>
                                                                ${clause.original_ambiguous_terms.map(term => `<span class="badge bg-warning text-dark">${term}</span>`).join(' ')}
                                                            </small>
                                                        </div>
                                                    ` : ''}
                                                </div>

                                                <div class="col-md-6">
                                                    <h6>✨ Claude Interpretation</h6>
                                                    <div class="bg-light p-3 rounded mb-2" style="max-height: 150px; overflow-y: auto;">
                                                        ${clause.interpreted_text || 'N/A'}
                                                    </div>
                                                    <div class="mb-2">
                                                        <small class="text-${clause.interpreted_ambiguous ? 'warning' : 'success'}">
                                                            ${clause.interpreted_ambiguous ? '⚠️ Still contains ambiguous terms' : '✅ No ambiguous terms remaining'}
                                                        </small>
                                                    </div>
                                                    ${clause.interpreted_ambiguous_terms && clause.interpreted_ambiguous_terms.length > 0 ? `
                                                        <div>
                                                            <small><strong>Remaining terms:</strong>
                                                                ${clause.interpreted_ambiguous_terms.map(term => `<span class="badge bg-warning text-dark">${term}</span>`).join(' ')}
                                                            </small>
                                                        </div>
                                                    ` : ''}
                                                </div>
                                            </div>

                                            <!-- Improvement Details -->
                                            ${clause.eliminated_terms && clause.eliminated_terms.length > 0 ? `
                                                <div class="alert alert-success">
                                                    <small><strong>✅ Successfully eliminated:</strong>
                                                        ${clause.eliminated_terms.map(term => `<span class="badge bg-success">${term}</span>`).join(' ')}
                                                    </small>
                                                </div>
                                            ` : ''}

                                            <!-- Metadata -->
                                            <div class="row mt-3">
                                                <div class="col-md-4">
                                                    <small><strong>Clause Type:</strong> ${clause.clause_type || 'unknown'}</small>
                                                </div>
                                                <div class="col-md-4">
                                                    <small><strong>Ambiguity Type:</strong> ${clause.ambiguity_type || 'N/A'}</small>
                                                </div>
                                                <div class="col-md-4">
                                                    <small><strong>Original Confidence:</strong> ${(clause.original_confidence || 0).toFixed(2)}</small>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    ` : ''}
                </div>
            `;
        }

        function getImprovementStatusText(status) {
            switch(status) {
                case 'improved': return 'Successfully Improved';
                case 'partial': return 'Partially Improved';
                case 'no_change': return 'No Change';
                default: return 'Unknown';
            }
        }

        function getImprovementStatusColor(status) {
            switch(status) {
                case 'improved': return 'success';
                case 'partial': return 'warning';
                case 'no_change': return 'secondary';
                default: return 'light';
            }
        }

        function getImprovementIcon(status) {
            switch(status) {
                case 'improved': return '✅';
                case 'partial': return '⚠️';
                case 'no_change': return '➖';
                default: return '❓';
            }
        }

        function getAdequacyColor(adequacy) {
            switch(adequacy.toLowerCase()) {
                case 'adequate': return 'success';
                case 'acceptable': return 'primary';
                case 'imprecise': return 'warning';
                case 'incomplete': return 'danger';
                default: return 'secondary';
            }
        }

        function getQualityColor(score) {
            if (score >= 0.8) return 'success';
            if (score >= 0.6) return 'primary';
            if (score >= 0.4) return 'warning';
            return 'danger';
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(MAIN_TEMPLATE, 
                                claude_enabled=bool(app_config.CLAUDE_API_KEY),
                                pipeline_ready=pipeline is not None)

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for contract analysis and smart contract generation"""
    if not pipeline:
        return jsonify({'success': False, 'error': 'Pipeline not initialized'}), 500
    
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        if not file or not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_filename = f"{timestamp}_{filename}"
        file_path = os.path.join('uploads', safe_filename)
        file.save(file_path)

        logger.info(f"Processing file: {safe_filename}")

        # Process contract (analysis and smart contract generation only)
        result = pipeline.process_contract(file_path)

        if result.success:
            # Prepare detailed response data with comparison
            analysis = result.contract_analysis
            comparison_data = {}

            # Extract comparison data if available
            if analysis and 'comparative_analysis' in analysis:
                comp_analysis = analysis['comparative_analysis']
                if 'clause_comparisons' in comp_analysis:
                    comparison_data = {
                        'clause_comparisons': comp_analysis['clause_comparisons'],
                        'aggregate_analysis': comp_analysis.get('aggregate_analysis', {}),
                        'improvement_stats': comp_analysis.get('improvement_stats', {}),
                        'adequacy': comp_analysis.get('adequacy', 'incomplete')
                    }

            # Prepare response data
            response_data = {
                'success': True,
                'analysis_summary': analysis.get('summary', {}) if analysis else {},
                'smart_contract_code': result.generated_contract_code,
                'processing_time': result.processing_time,
                'comparison_data': comparison_data,
                'detailed_clauses': analysis.get('transactional_clauses', []) if analysis else [],
                'interpretations': analysis.get('interpretations', []) if analysis else [],
                'bert_analysis': result.ambiguity_analysis
            }

            # Save results
            result_data = {
                'timestamp': datetime.now().isoformat(),
                'filename': filename,
                **response_data
            }
            
            result_file = os.path.join('pipeline_outputs', f'result_{timestamp}.json')
            with open(result_file, 'w') as f:
                json.dump(result_data, f, indent=2, default=str)
            
            return jsonify(response_data)
        else:
            logger.error(f"Analysis failed: {result.error_message}")
            return jsonify({'success': False, 'error': result.error_message}), 500
            
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'pipeline_ready': pipeline is not None,
        'claude_enabled': bool(app_config.CLAUDE_API_KEY)
    })

def cli_analysis(file_path: str, output_path: str = None, claude_api_key: str = None):
    """
    Command-line interface for contract analysis
    
    Args:
        file_path: Path to the contract file to analyze
        output_path: Path to save results (optional)
        claude_api_key: Claude API key for interpretations (optional)
    """
    if not pipeline:
        logger.error("Pipeline not initialized")
        return False
    
    try:
        logger.info(f"Analyzing contract: {file_path}")
        
        # Process contract
        result = pipeline.process_contract(file_path)
        
        if not result.success:
            logger.error(f"Analysis failed: {result.error_message}")
            return False
        
        # Print summary
        if result.contract_analysis and 'summary' in result.contract_analysis:
            summary = result.contract_analysis['summary']
            print(f"\n=== Analysis Results for {file_path} ===")
            print(f"Total Clauses: {summary.get('total_clauses', 0)}")
            print(f"Transactional Clauses: {summary.get('transactional_clauses', 0)}")
            print(f"AI Interpretations Generated: {summary.get('interpretations_generated', 0)}")
            print(f"Processing Time: {result.processing_time:.2f}s")

            # Display comparative analysis results
            if result.contract_analysis.get('comparative_analysis'):
                comp_analysis = result.contract_analysis['comparative_analysis']
                if 'aggregate_analysis' in comp_analysis:
                    agg = comp_analysis['aggregate_analysis']
                    successful = agg.get('successful_interpretations', 0)
                    avg_improvement = agg.get('average_improvement', 0.0)
                    total = agg.get('total_interpretations', 0)

                    print(f"\n=== Claude Interpretation Results ===")
                    print(f"Successful Interpretations: {successful}")
                    print(f"Total Interpretations: {total}")
                    print(f"Average Improvement: {avg_improvement:.1f}%")

                    if avg_improvement > 0:
                        print("✅ Claude interpretations improved contract clarity!")
                    elif avg_improvement < 0:
                        print("❌ Claude interpretations decreased clarity (need better prompting)")
                    else:
                        print("ℹ️  No measurable improvement from interpretations")
        
        # Save results if output path provided
        if output_path:
            result_data = {
                'success': True,
                'analysis_summary': result.contract_analysis.get('summary', {}) if result.contract_analysis else {},
                'smart_contract_code': result.generated_contract_code,
                'processing_time': result.processing_time,
                'bert_analysis': result.ambiguity_analysis
            }
            
            with open(output_path, 'w') as f:
                json.dump(result_data, f, indent=2, default=str)
            logger.info(f"Results saved to {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"CLI analysis failed: {e}")
        return False

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Legal Contract Ambiguity Analysis System')
    parser.add_argument('--cli', action='store_true', help='Run in CLI mode')
    parser.add_argument('--file', type=str, help='Contract file to analyze (CLI mode)')
    parser.add_argument('--output', type=str, help='Output file for results (CLI mode)')
    parser.add_argument('--api-key', type=str, help='Claude API key')
    
    args = parser.parse_args()
    
    if args.cli:
        if not args.file:
            print("Error: --file is required in CLI mode")
            sys.exit(1)
        
        # Run CLI analysis
        success = cli_analysis(
            file_path=args.file,
            output_path=args.output,
            claude_api_key=args.api_key
        )
        
        sys.exit(0 if success else 1)
    else:
        # Run web application
        logger.info("Starting Legal Contract Analysis System")
        
        if not app_config.CLAUDE_API_KEY:
            logger.warning("No Claude API key found. AI interpretations will be disabled.")
            logger.info("To enable AI interpretations, set the CLAUDE_API_KEY environment variable.")
        
        print("🚀 Starting Legal Contract Analysis Pipeline")
        print(f"📁 Upload folder: uploads/")
        print(f"🤖 Claude API: {'Enabled' if app_config.CLAUDE_API_KEY else 'Disabled (set CLAUDE_API_KEY in .env)'}")
        print(f"⚙️ Pipeline: {'Ready' if pipeline else 'Error - check logs'}")
        print("🌐 Open http://localhost:5000 in your browser")
        
        try:
            host = os.environ.get('HOST', '0.0.0.0')
            port = int(os.environ.get('PORT', 5000))
            debug = config_name == 'development'
            
            app.run(debug=debug, host=host, port=port)
            
        except KeyboardInterrupt:
            logger.info("Application stopped by user")
        except Exception as e:
            logger.error(f"Error running application: {e}")
            sys.exit(1) 