# Bureau of Ocean Energy Management Geospatial Data Viewer

A web application for visualizing and analyzing biodiversity data collected during aerial surveys of offshore wind energy areas.

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- Git

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/BOEM-webapp.git
cd BOEM-webapp
```

2. Create and activate a virtual environment:
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Environment Configuration

Create a `.env` file in the project root with the following configuration:
```bash
REPORT_SERVER_HOST=your_server_host
REPORT_SERVER_USER=your_username
REPORT_DIR=/path/to/reports/directory
```

Replace the values with your specific configuration:
- `REPORT_SERVER_HOST`: The host server for reports (e.g., hpg.rc.ufl.edu)
- `REPORT_SERVER_USER`: Your server username
- `REPORT_DIR`: Full path to the reports directory on the server

## 🚀 Running the Application

1. Ensure you're in the project directory with your virtual environment activated

2. Start the Streamlit app:
```bash
streamlit run app/main.py
```

3. The application will open in your default web browser at `http://localhost:8501`

## 📁 Project Structure

```
BOEM-webapp/
├── app/
│   ├── main.py              # Main application entry point
│   ├── pages/              
│   │   ├── Model_Development.py
│   │   ├── Observations.py
│   │   ├── Video.py
│   │   └── ...
│   └── utils/
│       └── vector_utils.py
├── .env                    # Environment configuration
├── .vscode/
│   └── launch.json        # VS Code debug configuration
├── prepare.py             # Data preparation script
├── requirements.txt       # Python dependencies
└── README.md
```

## 🛠️ Features

- Interactive visualization of biodiversity survey data
- Marine wildlife species detection and classification
- Distribution maps and abundance estimates
- Temporal and spatial pattern analysis
- Video playback of flight line footage
- Image galleries of detected species

## 🤝 Contributing

1. Fork the repository
2. Create a new branch for your feature
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 💬 Support

For support:
- Open an issue in the repository
- Contact the project maintainers

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Bureau of Ocean Energy Management
- University of Florida Research Computing
- Streamlit team
- All contributors and maintainers

