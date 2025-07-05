import logging
import yaml
from pathlib import Path
from bruit.preprocessing import preprocess
from bruit.preprocessing import feature_extraction
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
    feature_extraction_section = full_config.get("feature_extraction", {})

    npz_dir = resolve_path(path_config.trial_model_path)
         
    log_path = resolve_path(path_config.log_path)
    log_path.mkdir(parents=True, exist_ok=True)

    metadata_path = resolve_path(path_config.metadata_path)
    metadata_path.mkdir(parents=True, exist_ok=True)
    
    if config.model_training.classifier:
        # Iterate through each subfolder in the input directory
        for subfolder in input_path.iterdir():
            if not subfolder.is_dir():
                continue
            input_dir = resolve_parent_inserted_path(subfolder, input_path.parent)
            
            output_dir = resolve_path(path_config.preprocessed_audio_path) / subfolder.name
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save to new YAML file
            with open(metadata_path / "preprocessing.yaml", "w") as outfile:
                yaml.dump({"preprocessing": preprocessing_section}, outfile, default_flow_style=False)
            with open(metadata_path / "feature_extraction.yaml", "w") as outfile:
                yaml.dump({"preprocessing": feature_extraction_section}, outfile, default_flow_style=False)
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

            # Run the preprocessing
            preprocess.run_preprocessing(
                input_dir=input_dir,
                output_path=output_dir,
                config=full_config,
                quiet=quiet
            )

        logger.info("Starting feature extraction and saving to .npz files")
        segment_audio_dirs = []
        for subfolder in input_path.iterdir():
            if not subfolder.is_dir():
                continue
            output_dir = resolve_path(path_config.preprocessed_audio_path)
            segment_audio_path = output_dir / subfolder.name / "audio" / "segments"
            segment_audio_dirs.append(segment_audio_path.resolve())
        file_name = config.preprocessing.file_name
        file_extension = config.preprocessing.file_extension

        save_path = npz_dir / f"{file_name}.{file_extension}"
        feature_extraction.run_feature_extraction(
            segment_dirs=segment_audio_dirs,
            save_path=save_path,
            config=full_config,
            sr=preprocessing_section.get("sample_rate", 16000),
            quiet=quiet
        )

        
