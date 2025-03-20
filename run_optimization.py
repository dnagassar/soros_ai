# run_optimization.py
import schedule
import time
from modules.optimization import optimize_autogluon

def run_optimization_job():
    print("Starting hyperparameter optimization...")
    best_params, best_rmse = optimize_autogluon(n_trials=20)
    print(f"Optimization complete. Best RMSE: {best_rmse}, Best Parameters: {best_params}")

# Schedule the job to run daily at 02:00 (adjust as needed)
schedule.every().day.at("02:00").do(run_optimization_job)

if __name__ == "__main__":
    while True:
        schedule.run_pending()
        time.sleep(60)
