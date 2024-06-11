@echo off
timeout /t 10000 /nobreak

echo Starting Python scripts...

start "" cmd /k "call venv\Scripts\activate.bat && python src\main\create_dataset.py --compactness=0.03 --n_segments=2000 --downsample_factor=0.5"
start "" cmd /k "call venv\Scripts\activate.bat && python src\main\create_dataset.py --compactness=0.04 --n_segments=2000 --downsample_factor=0.5"
start "" cmd /k "call venv\Scripts\activate.bat && python src\main\create_dataset.py --compactness=0.05 --n_segments=2000 --downsample_factor=0.5"
start "" cmd /k "call venv\Scripts\activate.bat && python src\main\create_dataset.py --compactness=0.06 --n_segments=2000 --downsample_factor=0.5"
start "" cmd /k "call venv\Scripts\activate.bat && python src\main\create_dataset.py --compactness=0.07 --n_segments=2000 --downsample_factor=0.5"
start "" cmd /k "call venv\Scripts\activate.bat && python src\main\create_dataset.py --compactness=0.08 --n_segments=2000 --downsample_factor=0.5"
start "" cmd /k "call venv\Scripts\activate.bat && python src\main\create_dataset.py --compactness=0.09 --n_segments=2000 --downsample_factor=0.5"
start "" cmd /k "call venv\Scripts\activate.bat && python src\main\create_dataset.py --compactness=0.1 --n_segments=2000 --downsample_factor=0.5"

echo All scripts have been started.
