import os
from create_data_lists import create_data_lists
from create_supervise_mask import create_supervise_mask
from create_atlas import create_atlas

def setup_project():
    # Create necessary directories
    os.makedirs('list/MOTS', exist_ok=True)
    os.makedirs('snapshots/amos_ours_77', exist_ok=True)
    
    # Create data lists
    print("Creating data lists...")
    create_data_lists()
    
    # Create supervision mask
    print("Creating supervision mask...")
    create_supervise_mask()
    
    # Create atlas
    print("Creating atlas...")
    create_atlas()
    
    print("Setup complete!")

if __name__ == "__main__":
    setup_project() 