# PortfolioFrom13F

DB_NAME=
DB_USER=
DB_PASSWORD=
DB_HOST=
DB_PORT=

ETL Pipeline for processing 13F portfolio data with support for multiple data sources and destinations.

## Project Structure

```
PortfolioFrom13F/
├── ETL/
│   ├── DataHandlers/
│   │   ├── FilesDataHandler/
│   │   │   ├── csv_handler.py
│   │   │   └── xml_handler.py
│   │   └── DBDataHandler/
│   │       ├── db_handler.py
│   │       └── graph_db_handler.py
│   ├── DAL/
│   │   └── dal.py
│   ├── extractor.py
│   ├── manipulation.py
│   └── load.py
├── ETLPipeline.py
├── pipeline.py
├── requirements.txt
├── setup.sh
└── setup.ps1
```

## Setup

### 1. Clone/Navigate to Project
```bash
cd c:\Users\asafz\vscode\PortfolioFrom13F
```

### 2. Create Virtual Environment

**On Windows (PowerShell):**
```powershell
.\setup.ps1
```

**On macOS/Linux (Bash):**
```bash
chmod +x setup.sh
./setup.sh
```

**Manual Setup:**
```bash
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
pip install -r requirements.txt
```

## Running the Pipeline

### Example: CSV to CSV Pipeline

```python
from ETLPipeline import Pipeline
from ETL.extractor import CSVExtractor
from ETL.manipulation import TransformManipulation
from ETL.load import CSVLoad

# Define transformation logic
def transform_record(record):
    return {k: v.upper() if isinstance(v, str) else v for k, v in record.items()}

# Create pipeline
extractor = CSVExtractor('input.csv')
manipulation = TransformManipulation(transform_record)
load = CSVLoad('output.csv')

pipeline = Pipeline(extractor, manipulation, load)
pipeline.run()
```

### Example: Database to CSV Pipeline

```python
from ETLPipeline import Pipeline
from ETL.extractor import DatabaseExtractor
from ETL.manipulation import FilterManipulation
from ETL.load import CSVLoad
from ETL.data_handlers.db_data_handler.db_handler import DBHandler

# Create database handler
db_handler = DBHandler('localhost', 5432, 'mydb', 'user', 'password')


# Define filter criteria
def filter_criteria(record):
    return record.get('value', 0) > 1000


# Create pipeline
extractor = DatabaseExtractor('SELECT * FROM portfolio', db_handler)
manipulation = FilterManipulation(filter_criteria)
load = CSVLoad('filtered_output.csv')

pipeline = Pipeline(extractor, manipulation, load)
pipeline.run()
```

### Example: Chained Manipulations

```python
from ETLPipeline import Pipeline
from ETL.extractor import CSVExtractor
from ETL.manipulation import ChainManipulation, FilterManipulation, TransformManipulation
from ETL.load import CSVLoad

def filter_criteria(record):
    return record.get('quantity', 0) > 0

def transform_record(record):
    record['total'] = int(record.get('quantity', 0)) * float(record.get('price', 0))
    return record

# Chain multiple operations
chain = ChainManipulation([
    FilterManipulation(filter_criteria),
    TransformManipulation(transform_record)
])

extractor = CSVExtractor('input.csv')
load = CSVLoad('output.csv')

pipeline = Pipeline(extractor, chain, load)
pipeline.run()
```

## Deactivate Virtual Environment

```bash
deactivate
```

## Requirements

- Python 3.9+
- See `requirements.txt` for dependencies