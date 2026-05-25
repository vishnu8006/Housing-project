from src.preprocess import load_and_preprocess
from src.train import train_model
from src.evaluate import evaluate_model
def main():
 X_train, X_test, y_train, y_test = load_and_preprocess()
 model = train_model(X_train, y_train)
 rmse, r2 = evaluate_model(model, X_test, y_test)
 print("RMSE:", rmse)
 print("R2 Score:", r2)
if __name__ == "__main__":
 main()
 
