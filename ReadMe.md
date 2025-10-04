
# FrontEnd

In this `Readme.md` file, all the suggestions and required documentations will be here.

## Dataset Information

This project uses two CSV files:

- **`data_set.csv`** - Raw data file extracted directly from the database
- **`cleaned_data.csv`** - Cleaned and preprocessed version of the dataset, ready for analysis

## Setup Instructions

Follow these steps to set up and run the application:

### 1. Create a Virtual Environment

```bash
python -m venv .venv
```

### 2. Activate the Virtual Environment

**On Windows:**
```bash
.venv\Scripts\activate
```

**On macOS/Linux:**
```bash
source venv/bin/activate
```

### 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run main.py
```

## Requirements

Make sure you have Python installed on your system. All required libraries are listed in the `requirements.txt` file.

## Notes

- Ensure both CSV files are in the correct directory before running the application
- The virtual environment should be activated whenever you work on this project
