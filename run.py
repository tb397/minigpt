import subprocess
import sys
import os

def main():
    # Install Flask if needed
    try:
        import flask
    except ImportError:
        print("Installing Flask...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'flask'])

    # Launch the server
    print("Starting MiniGPT server at http://localhost:5000")
    os.system(f"{sys.executable} server.py")

if __name__ == '__main__':
    main()