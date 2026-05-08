.\.venv\Scripts\python.exe main.py

.\.venv\Scripts\python.exe frontend\server.py

# Face Attendance System

This project runs a live face-attendance detector and a local dashboard server.

## What each component does

- `setup.py` creates the SQLite database and required folders.
- `main.py` starts the live camera-based detection and attendance loop.
- `frontend/server.py` starts the local dashboard server.
- `frontend/index.html`, `frontend/app.js`, and `frontend/styles.css` render the dashboard UI.

## Requirements

- Windows
- Python 3.8 to 3.12
- A working camera

## One-time setup

Create and activate a virtual environment from the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the Python packages:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `pip install -r requirements.txt` fails on Windows because of `tensorflowjs`, install the runtime packages used by the app and continue:

```powershell
pip install opencv-python ultralytics mediapipe tensorflow face-recognition numpy Pillow
```

Initialize the database and folders:

```powershell
python setup.py
```

## Start the detection app

Run the live attendance detector:

```powershell
python main.py
```

The camera preview opens in a fullscreen OpenCV window. Press `Q` or `Esc` to quit.

## Start the dashboard server

Open a second terminal in the same virtual environment and run:

```powershell
python frontend/server.py
```

Then open:

```text
http://127.0.0.1:8000
```

## Typical run order

1. Activate the virtual environment.
2. Run `python setup.py` once.
3. Start `python main.py` in one terminal.
4. Start `python frontend/server.py` in another terminal.
5. Open the dashboard in your browser.

## Troubleshooting

- If the camera does not open, make sure no other app is using it and try changing `CAMERA_INDEX` in `main.py`.
- If the dashboard does not load, confirm the server is running on port `8000`.
- If dependency installation fails on Windows because of `tensorflowjs`, use the runtime install command above and keep the app running with the already-tested environment.