# ==========================================
# AI Project 2 - Data Classification Using AI
# Dataset: Iris Dataset
# Algorithm: K-Nearest Neighbors (KNN)
# ==========================================

# ---------- Import Libraries ----------

import tkinter as tk
from tkinter import messagebox

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score
)

import matplotlib.pyplot as plt


# ---------- Main Application Class ----------

class IrisClassifierApp:

    def __init__(self, root):
        self.root = root
        self.root.title("AI Project 2 - Iris Classification")
        self.root.geometry("650x500")
        self.root.resizable(False, False)

        self.cm = None

        # Title
        title = tk.Label(
            root,
            text="Data Classification Using AI",
            font=("Arial", 18, "bold")
        )
        title.pack(pady=10)

        # Subtitle
        subtitle = tk.Label(
            root,
            text="Iris Dataset Classification using KNN",
            font=("Arial", 12)
        )
        subtitle.pack()

        # K Value
        frame = tk.Frame(root)
        frame.pack(pady=15)

        tk.Label(
            frame,
            text="Enter K Value:"
        ).grid(row=0, column=0, padx=10)

        self.k_entry = tk.Entry(frame, width=10)
        self.k_entry.insert(0, "3")
        self.k_entry.grid(row=0, column=1)

        # Buttons
        train_btn = tk.Button(
            root,
            text="Train Model",
            width=20,
            command=self.train_model,
            bg="lightblue"
        )
        train_btn.pack(pady=5)

        cm_btn = tk.Button(
            root,
            text="Show Confusion Matrix",
            width=20,
            command=self.show_confusion_matrix,
            bg="lightgreen"
        )
        cm_btn.pack(pady=5)

        # Results Box
        self.result_text = tk.Text(
            root,
            height=15,
            width=75
        )
        self.result_text.pack(pady=15)

    # ---------- Train Model ----------

    def train_model(self):

        try:
            k = int(self.k_entry.get())

            # Load Dataset
            iris = load_iris()

            X = iris.data
            y = iris.target

            # Feature Scaling
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Split Dataset
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled,
                y,
                test_size=0.2,
                random_state=42
            )

            # Create Model
            model = KNeighborsClassifier(
                n_neighbors=k
            )

            # Train Model
            model.fit(X_train, y_train)

            # Predictions
            y_pred = model.predict(X_test)

            # Evaluation
            accuracy = accuracy_score(
                y_test,
                y_pred
            )

            f1 = f1_score(
                y_test,
                y_pred,
                average='weighted'
            )

            self.cm = confusion_matrix(
                y_test,
                y_pred
            )

            # Display Results
            self.result_text.delete(1.0, tk.END)

            self.result_text.insert(
                tk.END,
                "========== MODEL RESULTS ==========\n\n"
            )

            self.result_text.insert(
                tk.END,
                f"Dataset Name : Iris Dataset\n"
            )

            self.result_text.insert(
                tk.END,
                f"Total Samples : {len(X)}\n"
            )

            self.result_text.insert(
                tk.END,
                f"Training Samples : {len(X_train)}\n"
            )

            self.result_text.insert(
                tk.END,
                f"Testing Samples : {len(X_test)}\n\n"
            )

            self.result_text.insert(
                tk.END,
                f"K Value : {k}\n"
            )

            self.result_text.insert(
                tk.END,
                f"Accuracy Score : {accuracy:.2%}\n"
            )

            self.result_text.insert(
                tk.END,
                f"F1 Score : {f1:.4f}\n\n"
            )

            self.result_text.insert(
                tk.END,
                "Model Trained Successfully!"
            )

            messagebox.showinfo(
                "Success",
                "Model Trained Successfully!"
            )

        except ValueError:
            messagebox.showerror(
                "Error",
                "Please enter a valid K value."
            )

    # ---------- Confusion Matrix ----------

    def show_confusion_matrix(self):

        if self.cm is None:
            messagebox.showwarning(
                "Warning",
                "Train the model first!"
            )
            return

        plt.figure(figsize=(5, 4))
        plt.imshow(self.cm)

        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("Actual Label")

        for i in range(len(self.cm)):
            for j in range(len(self.cm)):
                plt.text(
                    j,
                    i,
                    str(self.cm[i][j]),
                    ha='center',
                    va='center'
                )

        plt.colorbar()
        plt.show()


# ---------- Run Application ----------

if __name__ == "__main__":

    root = tk.Tk()

    app = IrisClassifierApp(root)

    root.mainloop()