
import pathlib

def load_markdown_files(directory_path):
    docs=[]
    path = pathlib.Path(directory_path)
    
    if not path.exists() or not path.is_dir():
        raise ValueError(f"Invalid directory path: {directory_path}")
    
    for file_path in path.iterdir():
        if file_path.suffix in ['.txt', '.md']:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    if not content.strip():
                        print(f"Warning: {file_path} is empty.")
                    metadata = {
                        'file_name': file_path.name,
                        "source": str(file_path),
                    }
                    docs.append({"content": content, "metadata": metadata})
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    return docs

