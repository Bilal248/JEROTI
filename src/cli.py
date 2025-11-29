import os
import sys
import glob
import json
import time
import pandas as pd
from runner import run_detection_with_saved_model, stop_all, start_realtime_learning_and_detection, is_running
from data.collect import collect_and_save
from train import train_model


def print_main_menu():
    os.system('clear')
    print("\n==============================")
    print("\tJEROTI CLI     ")
    print("==============================")
    print("1. Manual options (train / collect)")
    print("2. Detect using trained model")
    print("3. Start / Stop Real-time detection with online learning")
    print("h. Help")
    print("q. Quit")
    print("==============================\n")


def print_help():
    os.system('clear')
    print("\n========== HELP ==========")
    print("1 - Manual options: train or collect data")
    print("2 - Stop background detection")
    print("3 - Detect using saved model")
    print("4 - Real-time detection with online learning")
    print("h - Show this help")
    print("q - Quit")
    print("==========================\n")


def print_model_perfornance(model):
    print(f"\tTotal Samples {model['total_samples']}")
    print(f"\tAnomalies Detected {model['anomalies_detected']}")
    print(f"\tAnomaly Fraction {model['anomaly_fraction']}")


def manual_menu():

    while True:
        os.system('clear')
        print("\n======= MANUAL MENU =======")
        print("1. Collect Data")
        print("2. Train new model")
        print("b. Back to main menu")
        print("===========================\n")

        choice = input("Choice: ").strip().lower()

        if choice == "1":
            seconds = input("Enter Seconds to Collect: ").strip()
            try:
                collect_and_save(duration=int(seconds))
            except:
                print("Running Default Seconds 15s.")
                collect_and_save(duration=int(15))
            input()

        elif choice == "2":
            os.system('clear')
            print("\nSelect model type:")
            print("1. Isolation Forest")
            print("2. DBSCAN")
            print("3. AUTO (choose best model)")
            print("4. Cancel Training\n")
            model_choice_input = input("Choice: ").strip()

            if model_choice_input == "1":
                model_choice = "isolation"
            elif model_choice_input == "2":
                model_choice = "dbscan"
            elif model_choice_input == "3":
                model_choice = "auto"
            elif model_choice_input == "4":
                continue
            else:
                print("Invalid choice.")
                continue

            file_input = input(
                "\nEnter file name(s) (comma-separated)\n"
                "or press ENTER to use all CSVs in data/: "
            ).strip()

            if file_input:
                file_paths = [
                    f"data/{f.strip()}.csv" for f in file_input.split(",")]
            else:
                file_paths = glob.glob("data/*.csv")

            if not file_paths:
                print("No files selected. Returning to menu.")
                continue

            try:
                df_list = [pd.read_csv(f) for f in file_paths]
                df = pd.concat(df_list, ignore_index=True)

                print(
                    f"\nTraining model using '{model_choice}' selection...")
                print(f"\n\tDatasets")
                fileindex = 1
                for file in file_paths:
                    print(f"\t{fileindex}). {file}")
                    fileindex = fileindex+1

                time.sleep(3)

                result = train_model(
                    df=df, dataset_names=file_paths, model_choice=model_choice)

                print("\n==== TRAINING RESULT ====")
                if model_choice == "auto":
                    print(f"Best model: {result['best_model']['model_name']}")
                    print(f"Model Performance:")
                    print_model_perfornance(
                        result['best_model']['performance'])

                else:
                    print(
                        f"Model saved: {result['selected_model']['model_name']}")
                    print(f"Model Performance:")
                    print_model_perfornance(
                        result['selected_model']['performance'])
                print("=========================\n")

                dummy = input("")

            except Exception as e:
                print(f"Error loading data or training model: {e}")
                dummy = input("")

        elif choice == "b":
            return
        else:
            print("Invalid option.")


def select_model():
    json_files = glob.glob("model/*.json")
    if not json_files:
        print("No models found in model directory")
        return None

    models = []
    for jf in json_files:
        with open(jf, "r") as f:
            models.append(json.load(f))

    print("\nAvailable Models:")
    for i, m in enumerate(models):
        perf = m.get("performance", {})
        print(
            f"{i+1}. {m['model_name']} | Total:{perf.get('total_samples')} "
            f"Anomalies:{perf.get('anomalies_detected')} "
            f"Fraction:{perf.get('anomaly_fraction')}"
        )

    choice = input("Select model number: ").strip()

    try:
        idx = int(choice) - 1
        selected = models[idx]

        print("\n--- Feature Stats ---")
        for col, stats in selected.get("feature_stats", {}).items():
            print(
                f"{col}: mean={stats['mean']:.2f}, "
                f"std={stats['std']:.2f}, "
                f"sample_z_scores={stats['z_scores_sample']}"
            )
        print("--------------------\n")

        return selected["model_file"]

    except:
        print("Invalid selection")
        return None


def main():
    while True:
        print_main_menu()
        choice = input("Enter choice: ").strip().lower()

        if choice == "1":
            manual_menu()

        elif choice == "2":
            model_path = select_model()
            if model_path:
                print("Enable ActionMode for saved model detection:")
                actionmode = int(
                    input("\tEnter 1 to kill anomalies, 2 to log only: "))
                if actionmode not in [1, 2]:
                    print("Invalid input. Defaulting to log only (2).")
                    actionmode = 2
                run_detection_with_saved_model(
                    model_path, actionmode=actionmode)

        elif choice == "3":
            if is_running():
                stop_all()  # Stop all learning + detection threads
                print("Real-time learning & detection stopped.")
            else:
                print("Enable ActionMode for real-time learning + detection:")
                actionmode = int(
                    input("\tEnter 1 to kill anomalies, 2 to log only: "))
                if actionmode not in [1, 2]:
                    print("Invalid input. Defaulting to log only (2).")
                    actionmode = 2

                print("Starting real-time learning & detection...")
                start_realtime_learning_and_detection(actionmode=actionmode)

        elif choice == "h":
            print_help()
            input("")

        elif choice == "q":
            if is_running():
                stop_all()
            print("Exiting...")
            sys.exit(0)

        else:
            print("Invalid choice. Retry...")


if __name__ == "__main__":
    main()