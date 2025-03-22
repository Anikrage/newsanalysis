import subprocess
import os

def run_fastapi():
    os.environ['PYTHONUNBUFFERED'] = '1'
    subprocess.Popen(['uvicorn', 'api:app', '--host', '0.0.0.0', '--port', '8000'])

def run_streamlit():
    os.environ['PYTHONUNBUFFERED'] = '1'
    subprocess.run(['streamlit', 'run', 'run.py', '--server.port=8501', '--server.address=0.0.0.0'])

if __name__ == '__main__':
    run_fastapi()
    run_streamlit()
