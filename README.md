# Algorithmic Beat Generator

This project is an algorithmic beat generator with a web-based interface. It allows users to generate unique beats using a combination of procedural audio synthesis and sample playback.

## How It Works
- The backend is a Python Flask server (`server.py`) that generates beats on request.
- The frontend is a single-page HTML client (`client.html`) that provides the user interface and interacts with the server.
- When a user requests a new beat, the client sends a request to the server, which generates a beat (using either built-in synthesis or available samples) and returns a downloadable WAV file.

## Files in This Repository
- **server.py**: The main backend server. Handles beat generation, audio processing, and serving the client.
- **client.html**: The web interface for users. Lets you generate, play, and download beats.

## Not Included
- **Sample Packs and Soundfonts**: Large sample folders and soundfont files are not included in this repository due to size. The server can synthesize all sounds if samples are missing, but you can add your own samples in a `samples/` folder for more variety.
- **Third-party Libraries**: You must install the required Python packages (see below).

## Requirements
- Python 3.8+
- Flask
- numpy
- scipy
- pydub

Install dependencies with:
```
pip install flask numpy scipy pydub
```

## Running the Project
1. Place `server.py` and `client.html` in the same directory.
2. (Optional) Add a `samples/` folder with your own drum, bass, and instrument samples for more variety.
3. Start the server:


Created by [Tonystathoulis](https://github.com/Tonystathoulis)