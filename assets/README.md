# Screenshots for Predictive Maintenance System

This directory contains screenshots and visual assets for the README.md documentation.

## Required Screenshots

1. **dashboard_overview.png** - Main Streamlit dashboard showing:
   - Statistics panel (Total Predictions, Risk Distribution, Avg Probability)
   - Risk gauge visualization
   - Recent predictions table
   - High-risk alerts section

2. **swagger_docs.png** - FastAPI Swagger UI showing:
   - POST /predict_failure endpoint with example
   - GET /health endpoint
   - Schema definitions
   - Interactive API testing interface

3. **model_performance.png** - Training output showing:
   - Cross-validation metrics
   - ROC-AUC scores
   - F2 threshold optimization
   - Confusion matrix visualization (if available)

4. **database_schema.png** - Database diagram showing:
   - All 5 tables with relationships
   - Foreign key constraints
   - Primary keys and indexes
   - Data flow between tables

## How to Add Screenshots

1. Run the system locally:
   ```bash
   # Start API
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   
   # Start Dashboard
   streamlit run dashboard/app.py --server.port 8501
   ```

2. Take screenshots:
   - Dashboard: http://localhost:8501
   - API Docs: http://localhost:8000/docs
   - Training output: Run `python -m model.train`
   - Database schema: Use DB visualization tool

3. Save images as PNG files in this directory with the names specified above

4. Optimize images for web (recommended max width: 1200px)

## Alternative: Placeholders

If screenshots aren't ready yet, the README will show broken image links but the structure is in place for easy updates.
