# Nutil Web UI

Web interface for running this module locally

## Setup

if you have not;

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

3. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

1. Upload your segmentation images (multiple files supported)
2. Upload the alignment file (JSON or WALN format)
3. Upload the atlas file (NRRD format)
4. Upload the label file (CSV format)
5. Configure the target color (RGB values)
6. Set the object cutoff threshold (optional)
7. Enable flat coordinates if needed (optional)
8. Click "Process Data"
9. Download the results as a ZIP file

## Requirements

- Python 3.9+
- Flask 3.0+
- Nutil library (parent directory)
