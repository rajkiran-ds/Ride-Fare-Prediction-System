# ============================================================
# RIDESENSE AI — MAIN ENTRY POINT
# Run: python main.py
# ============================================================

from src.ridesense_package.pipeline import run_pipeline


def main():

    print("Starting RideSense AI Pipeline...")

    run_pipeline()

    print("Pipeline Execution Completed Successfully.")


if __name__ == "__main__":

    main()