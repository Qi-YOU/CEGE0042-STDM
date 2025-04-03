# CEGE0042-STDM: New York City Taxi Trip Duration Data Analysis & Modelling

This project is based on the Kaggle competition ["NYC Taxi Trip Duration"](https://www.kaggle.com/competitions/nyc-taxi-trip-duration/overview). The competition provides a spatial-temporal dataset that records detailed information about taxi trips in New York City, including:

- **Pickup and dropoff locations** (latitude and longitude)  
- **Timestamps** for pickup and dropoff events  
- **Trip duration** (target variable to predict)  
- **Additional features** such as passenger count and vendor information  

## Objective  

The goal of the project is to analyze and model the taxi trip duration using spatio-temporal data. This involves exploring and visualizing patterns, trends, and relationships within the dataset, as well as implementing predictive modeling techniques to estimate trip durations.  

## System Requirements
The experiments were conducted on a system with the following specifications:
- GPU: Tesla T4
- CPU: 8-core Intel Xeon
- RAM: 32GB
- OS: Ubuntu 20.04 LTS
- Python: 3.8
- NVIDIA-SMI: 565.57.01
- GPU Driver Version: 565.57.01
- CUDA Version: 12.7  

## Installation Guide
### Prerequisites
1. Install [Anaconda](https://www.anaconda.com/products/distribution)
2. Ensure you have `unzip` installed (for handling data archives)
   ```bash
   sudo apt update && sudo apt install unzip
   ```
3. Access to the Ubuntu Terminal / Bash for command-line operations.
4. Verify NVIDIA-SMI Output  
   Run the following command:  

   ```bash
   nvidia-smi
   ```  

   should generate output similar to:  

   ```
   +-----------------------------------------------------------------------------------------+
   | NVIDIA-SMI 565.57.01              Driver Version: 565.57.01      CUDA Version: 12.7     |
   |-----------------------------------------+------------------------+----------------------+
   | GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
   | Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
   |                                         |                        |               MIG M. |
   |=========================================+========================+======================|
   |   0  Tesla T4                       On  |   00000000:00:06.0 Off |                  Off |
   | N/A   29C    P8              9W /   70W |       3MiB /  16384MiB |      0%      Default |
   |                                         |                        |                  N/A |
   +-----------------------------------------+------------------------+----------------------+

   +-----------------------------------------------------------------------------------------+
   | Processes:                                                                              |
   |  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
   |        ID   ID                                                               Usage      |
   |=========================================================================================|
   |  No running processes found                                                             |
   +-----------------------------------------------------------------------------------------+
   ```

   If your output differs significantly, particularly in CUDA version, driver version, or GPU detection, ensure:  

   - The **NVIDIA driver (565.57.01)** is installed correctly.  
   - The **CUDA version (12.7)** is properly recognized by `nvidia-smi`.  
   - The GPU is detected, and no errors are present.  

   If discrepancies persist, consider reinstalling the GPU drivers or updating CUDA.
### Step-by-Step Setup
1. **Clone the repository** or **Unzip the Code**:
   ```bash
   git clone https://github.com/Qi-YOU/CEGE0042-STDM.git
   cd CEGE0042-STDM
   ```
   or
   ```bash
   mkdir -p CEGE0042-STDM
   unzip code.zip -d CEGE0042-STDM/
   cd CEGE0042-STDM
   ```
2. **Create Conda Environment**  
   Create a Python 3.8 virtual environment named `nyc-taxi`:
   ```bash
   conda create --name nyc-taxi python=3.8
   ```
3. **Activate Conda Environment**
   ```
   conda activate nyc-taxi
   ```
4. **Install Dependencies**
    Install packages from requirements.txt using pip:
    ```bash
    pip install -r requirements.txt
    ```
5. **Install Jupyter Kernel**
    To run notebooks in the conda environment:
    ```bash
    python -m ipykernel install --user --name nyc-taxi --display-name "Python (nyc-taxi)"
    ```
6. **Launch Jupyter Notebook**
    ```bash
    jupyter notebook
    ```
    Select the nyc-taxi kernel when opening notebooks.
    1. After launching Jupyter, In the web interface:
       - Navigate to the notebook file
       - Click to open
       - Select "Python (nyc-taxi)" from the Kernel menu
       - Use Cell > Run All to execute entire notebook
    2. **Notebook Execution Order**:
        - **Exploratory Data Analysis (EDA)**:
          Open `eda.ipynb` and run all cells.
          
          *Runtime: 10-15 minutes on 8-core CPU*

        - **Autocorrelation Analysis**:
          Open `autocorrelation.ipynb` and run all cells.
          *Runtime: 20-30 minutes on 8-core CPU*

        - **Data Preparation**:
          Open `prep.ipynb` and run all cells.
          
          *Runtime: 3-5 hours on 8-core CPU. May take days on lower-spec hardware*

        - **Model Parameter Search**:
          Open `modeling-parameter-search.ipynb` and run all cells.

          *Runtime: ~16 hours on RTX 3080 GPU. CPU-only execution not recommended*

## Data Acknowledgements
This project utilizes data from multiple sources:

| File/Directory | Source |
|----------------|--------|
| `data/train.csv`, `data/test.csv` | [Kaggle NYC Taxi Trip Duration Competition](https://www.kaggle.com/competitions/nyc-taxi-trip-duration/data) |
| `utils/gz_2010_us_040_00_5m.json` | [US County GeoJSON (Kieran Healy)](https://github.com/kjhealy/us-county/blob/master/data/geojson/gz_2010_us_040_00_5m.json) |
| `utils/weather_data_nyc_centralpark_2016.csv` | [NYC Weather Data 2016 (Mathijs)](https://www.kaggle.com/datasets/mathijs/weather-data-in-new-york-city-2016) |
| `data/osrm/fastest_routes.parquet` | [NYC Taxi with OSRM (Oscar Leo)](https://www.kaggle.com/datasets/oscarleo/new-york-city-taxi-with-osrm) |
    
## Common Issues and Solutions

-  Conda Environment Activation Fails
   If activating your Conda environment fails, initialize Conda and restart your terminal:
   ```bash
   conda init
   ```
   After running this command, close and reopen your terminal.

-  Missing Packages in `requirements.txt`
   If some packages are missing or fail to install from `requirements.txt`, manually install them using Conda:
   ```bash
   conda install -c conda-forge <package_name>
   ```
   Replace `<package_name>` with the missing package's name.

- Out-of-Memory Errors
   If you experience memory issues while using Jupyter Notebook, launch it with increased buffer size:
   ```bash
   jupyter notebook --NotebookApp.max_buffer_size=1000000000
   ```
   Note: If you're using low-spec hardware or relying only on CPU, this may not resolve the issue and could make performance worse.

- Kernel Crashes (Kernel Killed or Died)
   If you frequently experience kernel crashes, try the following steps:
   1. Increase available system memory by closing unnecessary applications.
   2. Run Jupyter with a fresh environment:
      ```bash
      conda deactivate
      conda activate nyc-taxi
      jupyter notebook
      ```
   3. If the issue persists, reinstall Jupyter:
      ```bash
      conda install --force-reinstall jupyter
      ```
   Note: If you're using low-spec hardware or relying only on CPU, this may not resolve the issue.