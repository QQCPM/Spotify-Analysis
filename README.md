Music Recommendation Research System

A research-quality music recommendation system analyzing cross-cultural preferences (Vietnamese + Western music) with sophisticated temporal modeling. This project serves as foundational research for World Models, focusing on latent representation learning, temporal dynamics, and causal inference.

## Research Objectives

- **Latent Space Analysis**: Discover interpretable factors driving cross-cultural music preferences
- **Temporal Modeling**: Track and predict evolving musical tastes over time  
- **Cultural Bridge Discovery**: Identify gateway songs facilitating cross-cultural exploration
- **Causal Inference**: Establish causal relationships in preference formation and cultural adoption
- **World Models Preparation**: Build foundational understanding of sequential latent representations

## Project Structure

```
├── PROJECT_FRAMEWORK.md     # Comprehensive project documentation
├── src/                     # Core Python package
│   ├── data_processing/     # Spotify API, feature engineering
│   ├── models/             # Matrix factorization, neural models
│   ├── analysis/           # Cross-cultural and causal analysis
│   └── evaluation/         # Comprehensive evaluation framework
├── notebooks/              # Research Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_matrix_factorization.ipynb
│   ├── 03_temporal_analysis.ipynb
│   ├── 04_cultural_analysis.ipynb
│   └── 05_model_comparison.ipynb
├── tests/                  # Unit and integration tests
├── config/                 # Configuration files
├── results/                # Generated figures and analysis
└── docs/                   # Documentation and research papers
```

## Quick Start

### 1. Environment Setup
```bash
# Create conda environment
conda create -n music-research python=3.10
conda activate music-research

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### 2. Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your Spotify API credentials
# Get credentials at: https://developer.spotify.com/dashboard/
```

### 3. Run Data Exploration
```bash
# Launch Jupyter Lab
jupyter lab

# Open notebooks/01_data_exploration.ipynb
# Follow the research workflow through all 5 notebooks
```

## Research Methodology

This project follows rigorous research standards with:

- **Time-aware train/test splits** to prevent temporal data leakage
- **Statistical hypothesis testing** with proper power analysis
- **Comprehensive evaluation metrics** for accuracy, ranking, and cultural analysis
- **Reproducible results** with fixed random seeds and detailed logging
- **Open-source research platform** for community validation

## Development

### Code Quality
```bash
# Run tests with coverage
pytest --cov=src tests/

# Format code
black src/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ tests/
```

### Research Workflow
1. **Phase 1**: Data Foundation - Spotify collection and preprocessing
2. **Phase 2**: Matrix Factorization - Classical and temporal approaches  
3. **Phase 3**: Temporal Dynamics - LSTM and change point detection
4. **Phase 4**: Cross-Cultural Analysis - Bridge detection and causal inference
5. **Phase 5**: Evaluation & Research - Comprehensive analysis and publication

## Research Output

Expected deliverables:
- **5 Research Notebooks**: Complete analysis workflow
- **Technical Paper**: 8-12 pages with methodology and findings  
- **Open-Source Framework**: Production-ready recommendation system
- **Statistical Validation**: Rigorous hypothesis testing results

## Connection to World Models

This research builds foundational understanding for World Models through:
- **Latent Representations**: Matrix factorization as state space compression
- **Temporal Dynamics**: Sequential preference modeling as action sequences
- **Causal Reasoning**: Intervention analysis in preference formation

## Documentation

- See `PROJECT_FRAMEWORK.md` for comprehensive technical documentation
- Individual module documentation in `src/` directories
- Research methodology and findings in `notebooks/`

## Contributing

This is a research project. For questions or collaboration:
1. Read the comprehensive `PROJECT_FRAMEWORK.md` 
2. Review the research notebooks for methodology
3. Check existing issues and testing framework
4. Follow code quality standards (black, mypy, pytest)

## License

MIT License - See LICENSE file for details.

---

**Research Focus**: This project prioritizes research rigor, reproducibility, and methodological soundness over production deployment. All design decisions support the ultimate goal of advancing World Models research through cross-cultural temporal preference modeling.
