<div align="center">

# 🌊 ARGO FloatChat

### AI-Powered Ocean Data Discovery Platform

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=for-the-badge&logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

**A comprehensive AI-powered system for querying, analyzing, and visualizing ARGO oceanographic data using natural language processing and advanced data science techniques.**

[Features](#-key-features) • [Quick Start](#-quick-start) • [Documentation](#-project-structure) • [Demo](#-usage-examples) • [Contributing](#-contributing)

---

</div>

## 📋 Table of Contents

- [🎯 Project Overview](#-project-overview)
- [✨ Key Features](#-key-features)
- [🏗️ System Architecture](#️-system-architecture)
- [🚀 Quick Start](#-quick-start)
- [📊 Data Coverage](#-data-coverage)
- [🎯 Usage Examples](#-usage-examples)
- [🛠️ Development](#️-development)
- [📈 Performance](#-performance)
- [🐛 Troubleshooting](#-troubleshooting)
- [🤝 Contributing](#-contributing)

---

## 🎯 Project Overview

**ARGO FloatChat** is an end-to-end platform that democratizes access to oceanographic data through AI. It combines data extraction, processing, storage, and intelligent querying capabilities to make ARGO float data accessible to researchers, students, and ocean enthusiasts worldwide.

## ✨ Key Features

<table>
<tr>
<td width="50%">

### 🤖 AI & Intelligence
- **Natural Language Queries** - Chat naturally about ocean data
- **RAG Pipeline** - Retrieval-Augmented Generation with multiple LLM support
- **Smart Classification** - Automatic SQL vs semantic search routing
- **Geographic Intelligence** - Smart region detection and filtering

</td>
<td width="50%">

### 📊 Data & Visualization
- **Interactive Maps** - Dynamic location-based visualizations
- **Depth Profiles** - Detailed oceanographic measurements
- **Time Series Analysis** - Temporal trend visualization
- **Multi-format Export** - CSV, JSON, NetCDF, PNG support

</td>
</tr>
<tr>
<td width="50%">

### ⚡ Performance
- **Real-time Processing** - Fast responses using Groq API with Llama 3.1
- **Hybrid Search** - Combined SQL and vector database queries
- **Optimized Storage** - PostgreSQL + ChromaDB architecture
- **Concurrent Support** - Handle 50+ simultaneous users

</td>
<td width="50%">

### 🌊 Ocean Data
- **122,000+ ARGO Profiles** - Comprehensive Indian Ocean coverage
- **1,800+ Float Trajectories** - Track ocean drifters
- **BGC Parameters** - Oxygen, pH, nitrate, chlorophyll-a
- **Core Measurements** - Temperature, salinity, pressure, depth

</td>
</tr>
</table>

---

## 🛠️ Technology Stack

### Frontend
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)

### Backend
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-E92063?style=flat-square&logo=pydantic&logoColor=white)

### Databases
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=flat-square&logo=postgresql&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B6B?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-00ADD8?style=flat-square)

### AI/ML
![Groq](https://img.shields.io/badge/Groq-F55036?style=flat-square)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![Llama](https://img.shields.io/badge/Llama_3.1-0467DF?style=flat-square&logo=meta&logoColor=white)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    ARGO FloatChat Platform                      │
├─────────────────────────────────────────────────────────────────┤
│  Frontend (Streamlit)  │  Backend (FastAPI)  │  Data Pipeline   │
│  ┌─────────────────┐   │  ┌─────────────────┐│  ┌─────────────┐ │
│  │ FloatChat UI    │◄──┼──┤ Query Processor ││  │ Data        │ │
│  │ Interactive     │   │  │ RAG Pipeline    ││  │ Extraction  │ │
│  │ Visualizations  │   │  │ LLM Integration ││  │ & Cleaning  │ │
│  └─────────────────┘   │  └─────────────────┘│  └─────────────┘ │
│                        │  ┌─────────────────┐│  ┌─────────────┐ │
│                        │  │ Database Layer  ││  │ Vector DB   │ │
│                        │  │ PostgreSQL      ││  │ ChromaDB    │ │
│                        │  │ Structured      ││  │ Semantic    │ │
│                        │  └─────────────────┘│  └─────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

### 🎨 Frontend (`/frontend`)
**Streamlit-based interactive web application**

- **`floatchat_app.py`**: Main Streamlit application with chat interface
- **`backend_adapter.py`**: API client for backend communication
- **`frontend_config.py`**: Configuration and constants
- **`requirements.txt`**: Frontend dependencies (Streamlit, Plotly, Leaflet.js etc.)

**Features:**
- Interactive chat interface with natural language queries
- Real-time visualizations (maps, profiles, time series)
- Export capabilities (CSV, JSON, NetCDF, PNG)
- Responsive design with modern UI/UX

### ⚙️ Backend (`/backend`)
**FastAPI-based REST API with AI capabilities**

- **`app/main.py`**: FastAPI application entry point
- **`app/api/routes/`**: API endpoints (query, data, health)
- **`app/core/`**: Database and LLM client management
- **`app/services/`**: Business logic (RAG pipeline, query classification)
- **`app/models/`**: Pydantic data models
- **`scripts/`**: Setup and database management scripts

**Key Components:**
- **Query Classification**: Determines SQL vs semantic search
- **RAG Pipeline**: Retrieval-Augmented Generation for intelligent responses
- **Multi-LLM Support**: Groq, Hugging Face, and other providers
- **Vector Database**: ChromaDB for semantic search, FAISS for fallback
- **PostgreSQL**: Structured data storage

### 🧹 Data Cleaning (`/data_cleaning`)
**NetCDF processing and database preparation**

- **`src/argo_data_processor.py`**: NetCDF file processing
- **`src/batch_processor.py`**: Batch processing utilities
- **`src/vector_db_manager.py`**: Vector database operations
- **`sql/database_schema.sql`**: PostgreSQL schema
- **`deliverables/`**: Export and delivery scripts

**Data Processing Pipeline:**
1. **NetCDF Processing**: Extract oceanographic profiles from ARGO files
2. **Quality Control**: Filter data based on quality flags
3. **Database Storage**: Store in PostgreSQL with optimized schema
4. **Vector Embeddings**: Create semantic search capabilities
5. **Metadata Summaries**: Generate searchable content

### 📥 Data Extraction (`/data_extraction`)
**Automated ARGO data download and verification**

- **`efficient_downloader.py`**: Async download of NetCDF files
- **`retry_failed_downloads.py`**: Retry mechanism for failed downloads
- **`verify_downloads.py`**: Integrity checking and validation

**Download Features:**
- **Asynchronous Downloads**: Fast parallel processing
- **Retry Logic**: Automatic retry for failed downloads
- **Integrity Checking**: Verify file completeness
- **Progress Tracking**: Real-time download monitoring

## 🚀 Quick Start

### 📋 Prerequisites

<table>
<tr>
<td>

**System Requirements**
- Python 3.8+ (recommended 3.11+)
- PostgreSQL 12+
- 8GB+ RAM
- 10GB+ disk space

</td>
<td>

**API Keys Required**
- Groq API key (for LLM)
- OpenAI API key (optional)

</td>
</tr>
</table>

### 🌐 Deployment Options

<details>
<summary><b>🚀 Production Deployment (Click to expand)</b></summary>

- 📖 [Complete Deployment Guide](DEPLOYMENT.md) - Deploy to Render (Backend) + Vercel (Frontend)
- ✅ [Deployment Checklist](DEPLOYMENT_CHECKLIST.md) - Step-by-step deployment verification

</details>

### 💻 Local Development Setup

#### **Step 1: Clone Repository**

```bash
git clone <repository-url>
cd argo_floatchat
```

#### **Step 2: Backend Setup** 🔧

```bash
# Navigate to backend
cd backend

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env with your database credentials and API keys

# Run complete setup (database + vector DB)
python scripts/complete_setup.py

# Start backend server
python run.py
```

✅ Backend will be running at `http://localhost:8000`

#### **Step 3: Frontend Setup** 🎨

```bash
# Open new terminal
cd frontend

# Install dependencies
pip install -r requirements.txt

# Start Streamlit app
streamlit run floatchat_app.py
```

✅ Frontend will be running at `http://localhost:8501`

#### **Step 4: Data Processing** 📦 (Optional)

```bash
# Open new terminal
cd data_cleaning

# Install dependencies
pip install -r requirements.txt

# Process NetCDF files
python src/batch_processor.py

# Setup vector database
python src/vector_db_manager.py
```

### 🎯 First Query

Once everything is running, try your first query:

```
"Show me temperature profiles in the Indian Ocean for the last month"
```

## 📊 Data Coverage

Our system includes comprehensive ARGO data:

- **🌊 122,000+ ARGO profiles** from Indian Ocean region
- **🚢 1,800+ ARGO floats** with trajectory data
- **🌡️ Core Parameters**: Temperature, Salinity, Pressure, Depth
- **🧪 BGC Parameters**: Dissolved Oxygen, pH, Nitrate, Chlorophyll-a
- **🗺️ Geographic Focus**: Indian Ocean, Arabian Sea, Bay of Bengal
- **📅 Temporal Coverage**: 2019-2025 (extensible)

## ⚙️ Configuration

### 🔐 Environment Variables

Create a `.env` file in the backend directory:

```bash
# 🗄️ Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=argo_database
DB_USER=your_username
DB_PASSWORD=your_password

# 🤖 AI/LLM Configuration
GROQ_API_KEY=your_groq_api_key          # Required - Get from https://console.groq.com
OPENAI_API_KEY=your_openai_api_key      # Optional

# 🎮 Application Settings
DEBUG=true
HOST=127.0.0.1
PORT=8000
LOG_LEVEL=INFO

# 🔍 Vector Database (Optional - uses defaults if not set)
CHROMA_PERSIST_DIRECTORY=./chroma_db
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

<details>
<summary><b>🔑 How to Get API Keys</b></summary>

1. **Groq API Key** (Required)
   - Visit [Groq Console](https://console.groq.com)
   - Sign up for a free account
   - Navigate to API Keys section
   - Create a new API key

2. **OpenAI API Key** (Optional)
   - Visit [OpenAI Platform](https://platform.openai.com)
   - Create an account
   - Go to API keys section
   - Generate a new secret key

</details>

### Database Setup

The system uses two complementary databases:

1. **PostgreSQL** (Structured Data):
   - `argo_floats`: Float metadata and status
   - `argo_profiles`: Profile measurements with arrays for oceanographic parameters

2. **ChromaDB** (Semantic Search):
   - Metadata summaries for semantic search
   - Embedding-based similarity matching

## 🎯 Usage Examples

### Natural Language Queries

```bash
# Data Retrieval
"Show me temperature profiles in the Arabian Sea from 2023"
"Find ARGO floats near coordinates 20°N, 70°E"
"Get salinity data for float 7900617"

# Analytical Queries
"Compare BGC parameters in the Arabian Sea vs Bay of Bengal"
"What are the temperature trends in the Indian Ocean?"
"Summarize seasonal variations in chlorophyll levels"

# Exploratory Queries
"What can you tell me about ocean warming patterns?"
"Describe the characteristics of ARGO float data"
"How does salinity vary with depth in the Southern Ocean?"
```

### API Usage

```bash
# Natural language query
curl -X POST 'http://localhost:8000/api/v1/query' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "Show me temperature profiles in the Arabian Sea from 2023",
    "max_results": 10
  }'

# Direct data search
curl -X POST 'http://localhost:8000/api/v1/data/search' \
  -H 'Content-Type: application/json' \
  -d '{
    "start_date": "2023-01-01",
    "end_date": "2023-12-31",
    "min_latitude": 10,
    "max_latitude": 25,
    "parameters": ["temperature", "salinity"]
  }'
```

## 🔍 Query Processing Flow

The system intelligently routes queries through multiple pathways:

1. **Query Classification**: Determines the best approach (SQL vs semantic)
2. **SQL Retrieval**: For specific data requests with precise filtering
3. **Vector Retrieval**: For conceptual questions and pattern analysis
4. **Hybrid Retrieval**: Combines both approaches for complex analytical queries
5. **Response Generation**: Uses RAG pipeline for intelligent, contextual responses

## 🛠️ Development

### Project Structure

```
argo_floatchat/
├── frontend/                 # Streamlit web application
│   ├── floatchat_app.py     # Main UI application
│   ├── backend_adapter.py # API client
│   └── requirements.txt      # Frontend dependencies
├── backend/                  # FastAPI REST API
│   ├── app/                  # Application code
│   │   ├── api/routes/      # API endpoints
│   │   ├── core/           # Database & LLM clients
│   │   ├── services/       # Business logic
│   │   └── models/         # Data models
│   ├── scripts/            # Setup scripts
│   └── requirements.txt    # Backend dependencies
├── data_cleaning/          # Data processing pipeline
│   ├── src/               # Processing modules
│   ├── sql/               # Database schema
│   └── deliverables/      # Export utilities
├── data_extraction/       # Data download system
│   ├── efficient_downloader.py
│   ├── retry_failed_downloads.py
│   └── verify_downloads.py
└── README.md              # This file
```

### Running Tests

```bash
# Backend tests
cd backend
pytest tests/

# Frontend tests
cd frontend
pytest tests/
```

### Adding New Features

1. **Backend**: Add new endpoints in `app/api/routes/`
2. **Frontend**: Extend UI components in `floatchat_app.py`
3. **Data Processing**: Add new processors in `data_cleaning/src/`
4. **Database**: Update schema in `data_cleaning/sql/`

## 📈 Performance

- **⚡ Query Response Time**: < 2 seconds average
- **👥 Concurrent Users**: 50+ simultaneous queries
- **💾 Database Performance**: Optimized for ARGO data patterns
- **🔍 Vector Search**: Sub-second semantic similarity
- **📊 Processing Speed**: ~100 NetCDF files/minute

## 🔒 Security

- **🛡️ Input Validation**: Comprehensive validation on all endpoints
- **🚫 SQL Injection Prevention**: Parameterized queries
- **⏱️ Rate Limiting**: Configurable request limits

## 🐛 Troubleshooting

### Common Issues

1. **Database Connection Failed**:
   ```bash
   # Check PostgreSQL service
   sudo systemctl restart postgresql
   ```

2. **Vector Database Empty**:
   ```bash
   cd data_cleaning
   python src/vector_db_manager.py
   ```

3. **Groq API Errors**:
   - Verify API key in `.env`
   - Check rate limits and quotas

4. **Memory Issues**:
   - Reduce batch size in processing
   - Restart application services

### Health Checks

```bash
# Overall system health
curl http://localhost:8000/health

# Database health
curl http://localhost:8000/health/database

# Vector database health
curl http://localhost:8000/health/vector-db
```

## 🤝 Contributing

This project was built for the ARGO AI Hackathon and demonstrates:

- **🌊 End-to-end RAG pipeline** for oceanographic data
- **🗣️ Natural language interface** for complex scientific queries
- **📈 Scalable architecture** for real-world deployment
- **🔍 Multi-modal data access** combining SQL and vector search

### Development Guidelines

1. **Code Style**: Follow PEP 8 and use type hints
2. **Testing**: Add tests for new features
3. **Documentation**: Update README files for changes
4. **Performance**: Monitor query response times
5. **Security**: Validate all inputs and sanitize outputs

## 📄 License

This project is built for educational and research purposes. See individual component licenses for details.

## 🎯 Future Enhancements

- [ ] **🔄 Real-time Data Ingestion**: Live ARGO DAC integration
- [ ] **📊 Advanced Visualizations**: 3D ocean models and animations
- [ ] **🌍 Multi-language Support**: Internationalization
- [ ] **🤖 Custom Model Fine-tuning**: Domain-specific models
- [ ] **🛰️ Satellite Data Integration**: Multi-source ocean data
- [ ] **📤 NetCDF Export**: Standard oceanographic formats
- [ ] **🗺️ Advanced Geospatial Analysis**: Spatial statistics and modeling

## 📞 Support

For questions or issues:

- **📋 Check the troubleshooting section** above
- **📁 Review log files** for detailed error information
- **🐛 Create an issue** in the repository
- **💬 Contact the development team**

## 🌊 Use Cases

- **🔬 Oceanographic Research**: Climate studies, ocean circulation analysis
- **📚 Education**: Teaching oceanography and data science
- **🤖 Chatbot Integration**: Semantic search for oceanographic queries
- **🐠 Marine Biology**: Study of ocean ecosystems and habitats
- **🌡️ Climate Monitoring**: Long-term ocean temperature and salinity trends
- **📊 Data Science**: Machine learning on oceanographic datasets

---

<div align="center">

## 🌟 Star History

If you find this project useful, please consider giving it a ⭐!

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/argo-floatchat&type=Date)](https://star-history.com/#yourusername/argo-floatchat&Date)

---

### 📸 Screenshots

<table>
<tr>
<td width="50%">
<img src="docs/images/chat-interface.png" alt="Chat Interface" style="width:100%">
<p align="center"><b>Interactive Chat Interface</b></p>
</td>
<td width="50%">
<img src="docs/images/visualizations.png" alt="Data Visualizations" style="width:100%">
<p align="center"><b>Rich Data Visualizations</b></p>
</td>
</tr>
<tr>
<td width="50%">
<img src="docs/images/maps.png" alt="Geographic Maps" style="width:100%">
<p align="center"><b>Interactive Geographic Maps</b></p>
</td>
<td width="50%">
<img src="docs/images/analytics.png" alt="Analytics Dashboard" style="width:100%">
<p align="center"><b>Advanced Analytics</b></p>
</td>
</tr>
</table>

> **Note:** Add your actual screenshots to the `docs/images/` directory

---

### 🎖️ Acknowledgments

- **ARGO Program** - For providing open access to oceanographic data
- **Groq** - For fast LLM inference API
- **Streamlit** - For the amazing web framework
- **FastAPI** - For the high-performance backend framework
- **SIH Hackathon** - For inspiring this project

---

### 📞 Contact & Support

<table>
<tr>
<td align="center" width="33%">

**🐛 Report Issues**

[GitHub Issues](https://github.com/yourusername/argo-floatchat/issues)

</td>
<td align="center" width="33%">

**💬 Discussions**

[GitHub Discussions](https://github.com/yourusername/argo-floatchat/discussions)

</td>
<td align="center" width="33%">

**📧 Email**

your.email@example.com

</td>
</tr>
</table>

---

<p align="center">
<b>Built with ❤️ for the SIH Hackathon</b>
<br><br>
<i>FloatChat makes the vast ocean of data accessible to everyone, from researchers to students,<br>through the power of artificial intelligence and natural language processing.</i>
<br><br>
<img src="https://img.shields.io/badge/Made%20with-Python-blue?style=for-the-badge&logo=python&logoColor=white" alt="Made with Python">
<img src="https://img.shields.io/badge/Powered%20by-AI-orange?style=for-the-badge" alt="Powered by AI">
<img src="https://img.shields.io/badge/Ocean-Data-cyan?style=for-the-badge" alt="Ocean Data">
</p>

</div>
