import zipfile
import os

def zip_project(output_filename="colab_package.zip"):
    paths_to_zip = [
        "finrl_pro",
        "scripts",
        "configs",
        "pyproject.toml",
        "setup.py",
        "data/sp500_phase9_2005_2025.parquet",
        "results/phase9_sonnet_protocol/phase9_results.csv"
    ]
    
    print(f"Creating {output_filename}...")
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for path in paths_to_zip:
            if os.path.isfile(path):
                zipf.write(path)
                print(f"Added: {path}")
            elif os.path.isdir(path):
                for root, dirs, files in os.walk(path):
                    for file in files:
                        # Ignore python cache and git
                        if "__pycache__" in root or ".git" in root or ".DS_Store" in file:
                            continue
                        file_path = os.path.join(root, file)
                        zipf.write(file_path)
                        # print(f"Added: {file_path}") # verbose
                print(f"Added Folder: {path}")
    
    print(f"Package created successfully at: {os.path.abspath(output_filename)}")

if __name__ == "__main__":
    zip_project()
