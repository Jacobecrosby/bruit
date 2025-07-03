import logging
import yaml
from pathlib import Path
#from bruit.preprocessing import audioPreprocessor
from bruit.modules.config_loader import load_config, resolve_path, resolve_parent_inserted_path

def run_preprocessing(input_dir, config=None, quiet=False):
    path_config = load_config("configs/paths.yaml")
    input_path = Path(input_dir)

    # Load the original YAML
    with open("configs/parameters.yaml", "r") as infile:
        full_config = yaml.safe_load(infile)
     # Extract the 'preprocessing' section
    preprocessing_section = full_config.get("preprocessing", {})
    plotting_section = full_config.get("plotting", {})
    
    if config.model_training.classifier:
      
        npz_dir = resolve_path(path_config.component_model_path)

        log_path = resolve_path(path_config.log_path)
        log_path.mkdir(parents=True, exist_ok=True)

        metadata_path = resolve_path(path_config.metadata_path)
        metadata_path.mkdir(parents=True, exist_ok=True)
        
        # Save to new YAML file
        with open(metadata_path / "preprocessing.yaml", "w") as outfile:
            yaml.dump({"preprocessing": preprocessing_section}, outfile, default_flow_style=False)
        with open(metadata_path / "plotting.yaml", "w") as outfile:
            yaml.dump({"plotting": plotting_section}, outfile, default_flow_style=False)
        
        # Logging setup
        logging.basicConfig(
            filename=log_path / "preprocessing.log",
            filemode="w",
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        )
        logger = logging.getLogger("bruit")
        if not quiet:
            logger.info(f"📁 Logging all preprocessing steps to {log_path}")
