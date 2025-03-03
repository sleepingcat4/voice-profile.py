import os
import pandas as pd
import subprocess
from tqdm import tqdm

def process_audio_files(source_folder, target_folder, csv_file, output_dir, checkpoint_path, config_path):
    names_df = pd.read_csv(csv_file)
    names = names_df['name'].tolist()
    
    target_files = [f for f in os.listdir(target_folder) if f.endswith('.wav')]
    source_files = [f for f in os.listdir(source_folder) if f.endswith('.wav')]
    
    target_index = 0
    name_index = 0
    
    while target_index < len(target_files) and name_index < len(names):
        target_file = target_files[target_index]
        target_path = os.path.join(target_folder, target_file)
        
        name = names[name_index]
        output_name_dir = os.path.join(output_dir, name)
        os.makedirs(output_name_dir, exist_ok=True)
        
        for source_file in tqdm(source_files, desc=f"Processing Source Files for {target_file}"):
            source_path = os.path.join(source_folder, source_file)
            output_file = os.path.join(output_name_dir, f"{name}_{source_file}")
            
            command = [
                "python3", "inference.py", 
                "--source", source_path, 
                "--target", target_path, 
                "--output", output_file, 
                "--diffusion-steps", "50", 
                "--length-adjust", "1.0", 
                "--inference-cfg-rate", "0.7", 
                "--config-path", config_path, 
                "--checkpoint-path", checkpoint_path
            ]
            
            subprocess.run(command)
        
        target_index += 1
        name_index += 1

source_folder = '/media/results/11lab/rname'
target_folder = '/media/results/11lab/11lab-voice'
csv_file = '/media/results/11lab/names.csv'
output_dir = '/media/results/11lab/profile'
checkpoint_path = '/media/results/11lab/model/ft_model.pth'
config_path = '/media/results/11lab/model/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml'

process_audio_files(source_folder, target_folder, csv_file, output_dir, checkpoint_path, config_path)
