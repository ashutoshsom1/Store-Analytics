# IKEA Store Analytics 🏪

An AI-powered business analytics dashboard for IKEA retail operations in the UAE region. This Streamlit application provides comprehensive data analysis and insights for DJA and YAS stores, combining natural language querying with advanced analytics capabilities.

## 🎯 Overview

IKEA Store Analytics is an intelligent assistant that helps analyze store performance data through natural language queries. It leverages AI agents, vector databases, and interactive visualizations to provide business insights for IKEA retail operations.

### Key Features

- **Natural Language Querying**: Ask questions about store performance in plain English
- **Multi-Store Analysis**: Compare performance between DJA and YAS stores
- **Channel Analytics**: Analyze both Store and IFB (IKEA Food & Beverages) channels
- **Interactive Visualizations**: Dynamic charts and graphs using Streamlit
- **AI-Powered Insights**: LangChain agents with fallback mechanisms
- **Real-time Data Processing**: Vector similarity search for relevant data retrieval
- **Executive Analytics**: Comprehensive KPI analysis and reporting

## 📊 Key Performance Indicators (KPIs)

The system tracks and analyzes the following metrics:

- **Actuals (Act)**: Net revenue in AED (primary metric)
- **Last Year (Ly)**: Previous year comparison data
- **vs Ly%**: Year-over-year percentage change
- **Visitors**: Total footfall count
- **Customers**: Number of completed transactions
- **Conversion Rate**: Customer conversion (Customers/Visitors)
- **ATV**: Average Transaction Value (Revenue/Customers)
- **Items Sold**: Total quantity of items sold
- **Price/Item**: Average price per item
- **Item/Customer**: Average items per transaction

## 🏗️ Project Structure

```
IKEA/
├── v1.0.0/
│   ├── app.py                      # Main Streamlit application
│   ├── assistant.py                # Assistant interface
│   ├── main.py                     # Core query processing logic
│   ├── month_growth_analysis.py    # Monthly growth analysis utilities
│   ├── requirements.txt            # Python dependencies
│   ├── logger.py                   # Logging configuration
│   │
│   ├── configuration/
│   │   ├── config.py              # Application configuration
│   │   └── ikea_store_prompts.yaml # AI agent prompts and instructions
│   │
│   ├── constants/
│   │   └── constant.py            # Application constants
│   │
│   ├── data/
│   │   ├── dataloader.py          # Data loading and vector DB setup
│   │   ├── ikea_store_data.csv    # Main store data
│   │   ├── ikea_store_database.db # SQLite database
│   │   ├── ikea_sql_vectors.faiss # Vector embeddings
│   │   ├── ikea_metadata.pkl      # Metadata cache
│   │   ├── datasets/              # Raw data files
│   │   ├── documents/             # PDF documentation
│   │   └── retriever/             # Vector retrieval indices
│   │
│   ├── tools/
│   │   ├── tools_agents.py        # Main AI agents
│   │   └── fallback_agents.py     # Fallback AI agents
│   │
│   ├── utils/
│   │   ├── llm_setup.py          # LLM configuration
│   │   ├── tools.py              # Utility tools
│   │   ├── utils.py              # General utilities
│   │   ├── executive_analytics.py # Executive reporting
│   │   └── fallback_tools.py     # Fallback utilities
│   │
│   └── logs/
│       └── app.log               # Application logs
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager
- Access to Azure OpenAI services (optional, based on configuration)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ashutoshsom1/IKEA.git
   cd IKEA/v1.0.0
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Setup**
   Create a `.env` file in the root directory with your configuration:
   ```env
   # Add your API keys and configuration here
   OPENAI_API_KEY=your_openai_key
   AZURE_OPENAI_ENDPOINT=your_azure_endpoint
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Access the dashboard**
   Open your browser and navigate to `http://localhost:8501`

## 💡 Usage Examples

### Natural Language Queries

Ask questions like:
- "What was the total revenue for DJA store in January 2025?"
- "Compare conversion rates between DJA and YAS stores"
- "Show me the month-over-month growth trends"
- "Which store performed better in Q1 2025?"
- "What's the average transaction value for IFB channel?"

### Interactive Features

- **Chat Interface**: Type your questions in natural language
- **Visual Analytics**: Automatic chart generation for data insights
- **Store Comparison**: Side-by-side performance analysis
- **Time Series Analysis**: Trend analysis over different periods
- **Executive Reports**: Comprehensive business summaries

## 🔧 Configuration

### AI Agent Configuration

The system uses configurable AI agents defined in `configuration/ikea_store_prompts.yaml`:

- **Base Instructions**: Core behavior and data context
- **Fallback Mechanisms**: Handling edge cases and errors
- **Query Processing**: SQL generation and execution guidelines
- **Visualization Rules**: Chart generation parameters

### Database Schema

The SQLite database contains store performance data with the following structure:
- **Region**: AE (United Arab Emirates)
- **Stores**: DJA, YAS
- **Channels**: Store, IFB
- **Time Period**: January - June 2025
- **Currency**: AED (UAE Dirham)

## 📈 Analytics Capabilities

### Store Performance Analysis
- Revenue tracking and comparison
- Visitor and customer analytics
- Conversion rate optimization
- Average transaction value analysis

### Trend Analysis
- Month-over-month growth
- Year-over-year comparisons
- Seasonal pattern identification
- Anomaly detection

### Channel Analysis
- Store vs IFB performance
- Channel-specific KPI tracking
- Cross-channel insights

## 🛠️ Technical Architecture

### Core Components

- **Streamlit Frontend**: Interactive web interface
- **LangChain Agents**: AI-powered query processing
- **FAISS Vector Store**: Similarity search for data retrieval
- **SQLite Database**: Structured data storage
- **Matplotlib/Streamlit Charts**: Data visualization

### AI Pipeline

1. **Query Processing**: Natural language understanding
2. **SQL Generation**: Automatic query creation
3. **Data Retrieval**: Vector similarity search
4. **Result Processing**: Data analysis and formatting
5. **Visualization**: Chart generation and display

## 📝 Logging

The application includes comprehensive logging:
- Query processing logs
- Error handling and debugging
- Performance metrics
- User interaction tracking

Logs are stored in `logs/app.log` with configurable levels.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📄 License

This project is proprietary software for IKEA retail operations.

## 🆘 Support

For support and questions:
- Check the logs in `logs/app.log`
- Review the configuration in `configuration/`
- Contact the development team

## 📚 Documentation

Additional documentation includes:
- `documents/IKEA_Store_Operations.pdf`
- `documents/IKEA_Performance_Guidelines.pdf`
- `documents/IKEA_Food_Beverages.pdf`

## 🔄 Version History

### v1.0.0
- Initial release
- Basic store analytics functionality
- AI-powered query processing
- Interactive Streamlit dashboard
- Multi-store comparison capabilities

---

**Built with ❤️ for IKEA Retail Analytics**
