import sys
import os

# Add the root directory to the python path to load the src module
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.pipeline import preprocess_german_credit, preprocess_gmsc

def main():
    print("==========================================================")
    print("Shap Synthetic Credit Risk: Starting Preprocessing Pipeline")
    print("==========================================================\n")
    
    # 1. Preprocess German Credit
    try:
        preprocess_german_credit()
    except Exception as e:
        print(f"Error during German Credit preprocessing: {e}\n")
        
    # 2. Preprocess GMSC
    try:
        preprocess_gmsc()
    except FileNotFoundError as e:
        print(f"\n[ACTION REQUIRED] GMSC Preprocessing Skipped:")
        print(e)
        print("\nPlease download the dataset and place it in the raw folder to complete GMSC preprocessing.")
    except Exception as e:
        print(f"Error during GMSC preprocessing: {e}\n")
        
    print("\n==========================================================")
    print("Preprocessing Script Execution Completed")
    print("==========================================================")

if __name__ == "__main__":
    main()
