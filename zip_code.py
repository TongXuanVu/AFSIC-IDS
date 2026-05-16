import os
import zipfile

def zip_code(output_filename, source_dir):
    # Cac thu muc va file can loai bo
    exclude_dirs = {
        'logs', 
        'data', 
        '.git', 
        '__pycache__', 
        '.vscode', 
        '.ipynb_checkpoints',
        'checkpoints',
        'checkpoints_rounds'
    }
    exclude_files = {
        output_filename, 
        'zip_code.py', 
        'SPCIL.pdf', 
        'ciciot23_fl_split.zip',
        'HFIN_code_kaggle.zip'
    }
    exclude_extensions = {'.pyc', '.pyo', '.pyd', '.pth', '.pkl', '.npy', '.csv', '.png', '.log'}

    print(f"Starting to zip {source_dir} into {output_filename}...")
    
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            # Loai bo cac thu muc khong can thiet khoi danh sach duyet
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                if file in exclude_files:
                    continue
                if any(file.endswith(ext) for ext in exclude_extensions):
                    # Ngoai le: giu lai cac file .csv neu no nam trong exps/ (neu co)
                    # Nhung thuong exps chi chua .json
                    continue
                
                file_path = os.path.join(root, file)
                # Lay duong dan tuong doi de luu vao zip
                arcname = os.path.relpath(file_path, source_dir)
                
                zipf.write(file_path, arcname)
                print(f"  Added: {arcname}")

    print(f"\nSuccessfully created {output_filename}")

if __name__ == "__main__":
    # Ten file output
    zip_name = "SPCIL_kaggle.zip"
    # Duong dan thu muc hien tai
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    zip_code(zip_name, current_dir)
