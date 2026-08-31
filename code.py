import subprocess
import sys
import importlib

# Dictionary mapping: { "import_name": "pip_install_name" }
libraries = {
    "streamlit": "streamlit",
    "pandas": "pandas",
    "plotly": "plotly",
    "numpy": "numpy",
    "scipy": "scipy",
    "ydata_profiling": "ydata-profiling",
    "sklearn": "scikit-learn",
    "openpyxl": "openpyxl",
    "pyarrow": "pyarrow"
}

def install_and_check():
    print("🔍 Starting library check...\n")
    
    for import_name, install_name in libraries.items():
        try:
            importlib.import_module(import_name)
            print(f"✅ {import_name} is already installed.")
        except ImportError:
            print(f"⚠️  {import_name} not found. Installing {install_name}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", install_name])
                print(f"Successfully installed {install_name}!\n")
            except Exception as e:
                print(f"❌ Failed to install {install_name}: {e}")

    print("\n✨ All checks complete! You can now run your app using:")
    print("streamlit run your_filename.py")

if __name__ == "__main__":
    install_and_check()