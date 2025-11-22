import os
import pandas as pd
import json
from huggingface_hub import hf_hub_download

def main():
    repo_id = "nkp37/OpenVid-1M"
    filename = "data/train/OpenVid-1M.csv"
    
    print(f"Downloading {filename} from {repo_id}...")
    try:
        file_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        print(f"Downloaded to: {file_path}")
    except Exception as e:
        print(f"Error downloading file: {e}")
        return

    print("Loading CSV...")
    try:
        df = pd.read_csv(file_path)
        print(f"Columns found: {df.columns.tolist()}")
        
        # Identify caption column
        caption_col = None
        for col in ['caption', 'text', 'description']:
            if col in df.columns:
                caption_col = col
                break
        
        if caption_col:
            print(f"Using column '{caption_col}' for captions.")
            captions = df[caption_col].dropna().head(10000).tolist()
            
            output_file = "captions_10k.json"
            with open(output_file, "w") as f:
                json.dump(captions, f, indent=2)
            
            print(f"Successfully saved {len(captions)} captions to {output_file}")
        else:
            print("Could not find a suitable caption column. Please check the column names above.")
            
    except Exception as e:
        print(f"Error processing CSV: {e}")

if __name__ == "__main__":
    main()
