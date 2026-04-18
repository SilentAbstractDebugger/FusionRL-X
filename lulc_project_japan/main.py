import argparse
import logging
import sys
import traceback
from sklearn.model_selection import train_test_split 
# Added for proper ML evaluation

# Assuming these are your custom modules
from src.config import Config
from src.data_acquisition import download_landsat_scene
from src.cloud_masking import apply_cloud_mask
from src.preprocessing import preprocess_landsat, create_patches
from src.ndvi import compute_ndvi
from src.model import build_unet, build_baseline_cnn
from src.train import train_model
from src.evaluation import evaluate_model
from src.visualization import (
    plot_true_color,
    plot_ndvi,
    plot_lulc,
    plot_confusion_matrix
)

# Streamlit app is separate, not executed here

def setup_logger() -> logging.Logger:
    """Configures a standard logger for the pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("pipeline_execution.log") 
            |# Saves logs to a file
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logger()

def run_pipeline(args: argparse.Namespace) -> None:
    logger.info("=========================================")
    logger.info(" STARTING LULC PIPELINE FOR JAPAN AOI ")
    logger.info("=========================================")

    try:
        # STEP 1: DOWNLOAD DATA
        logger.info("Step 1: Downloading Landsat data...")
        scene_path = download_landsat_scene(
            aoi_path=Config.AOI_PATH,
            max_cloud_cover=Config.MAX_CLOUD_COVER
        )

        # STEP 2: CLOUD MASKING
        logger.info("Step 2: Applying cloud mask...")
        masked_scene = apply_cloud_mask(scene_path)

        # STEP 3: PREPROCESSING
        logger.info("Step 3: Preprocessing data...")
        stacked_image = preprocess_landsat(masked_scene)

        # STEP 4: NDVI
        logger.info("Step 4: Computing NDVI...")
        ndvi = compute_ndvi(stacked_image)

        # VISUALIZATION (True Color + NDVI)
        logger.info("Generating visual outputs...")
        plot_true_color(stacked_image)
        plot_ndvi(ndvi)

        # STEP 5: LOAD LULC LABELS
        logger.info("Step 5: Loading ESRI LULC data...")
        lulc_labels = Config.load_lulc()
        plot_lulc(lulc_labels)

        # STEP 6: PATCH EXTRACTION
        logger.info("Step 6: Creating training patches...")
        X, y = create_patches(
            stacked_image,
            ndvi,
            lulc_labels,
            patch_size=Config.PATCH_SIZE
        )

        # CRITICAL FIX: Train/Test Split to prevent data leakage
        logger.info("Step 6.5: Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # STEP 7: MODEL SELECTION
        logger.info(f"Step 7: Building {args.model.upper()} model...")
        if args.model == "unet":
            model = build_unet(input_shape=X_train.shape[1:])
        else:
            model = build_baseline_cnn(input_shape=X_train.shape[1:])

        # STEP 8: TRAINING
        logger.info("Step 8: Training model...")
        history = train_model(
            model,
            X_train,
            y_train,
            epochs=Config.EPOCHS,
            batch_size=Config.BATCH_SIZE
        )

        # Save model
        model.save(Config.MODEL_PATH)
        logger.info(f"Model successfully saved at {Config.MODEL_PATH}")

        # STEP 9: EVALUATION
        logger.info("Step 9: Evaluating model on TEST data...")
        metrics = evaluate_model(model, X_test, y_test)

        plot_confusion_matrix(metrics["confusion_matrix"])

        logger.info("=== FINAL RESULTS ===")
        logger.info(f"F1 Score:  {metrics.get('f1_score', 'N/A')}")
        logger.info(f"IoU Score: {metrics.get('iou_score', 'N/A')}")
        logger.info(f"Accuracy:  {metrics.get('accuracy', 'N/A')}")
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")

    except Exception as e:
        logger.error(f"Pipeline failed due to an error: {str(e)}")
        logger.debug(traceback.format_exc()) # Captures the exact line of the crash
        sys.exit(1) # Exit with a failure status code


def main() -> None:
    parser = argparse.ArgumentParser(description="LULC Classification Pipeline")

    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute the full machine learning pipeline"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="unet",
        choices=["unet", "cnn"],
        help="Specify the neural network architecture to build (default: unet)"
    )

    args = parser.parse_args()

    if args.run:
        run_pipeline(args)
    else:
        # Standardize missing argument response
        parser.print_help()
        print("\nNotice: Please run the script with the --run flag to execute the pipeline.")


if __name__ == "__main__":
    main()
