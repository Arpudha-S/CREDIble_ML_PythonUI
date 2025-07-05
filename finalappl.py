import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, roc_curve
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Global variables
data = None
X_train, X_test, y_train, y_test = None, None, None, None
scaler = StandardScaler()
models = {
    "Logistic Regression": None,
    "Random Forest": None,
    "XGBoost": None,
    "Isolation Forest": None,
}
filterData = None
tree = None
figure = None
canvas = None
total_spent_label = None
warning_label = None

# Existing functions (unchanged)
def load_data():
    global data
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, "Loading...")
    result_text.pack(padx=1, pady=1)
    filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if filepath:
        try:
            data = pd.read_csv(filepath)
            result_text.pack_forget()
            messagebox.showinfo("Success", "Dataset loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {e}")
    else:
        messagebox.showwarning("Warning", "No file selected!")

def preprocess_data():
    global data, X_train, X_test, y_train, y_test
    if data is None:
        messagebox.showwarning("Warning", "Please load the dataset first!")
        return

    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, "Preprocessing, Please wait...")
    result_text.pack(padx=1, pady=1)
    result_frame.update()

    columns_to_drop = [
        'trans_date_trans_time', 'first', 'last', 'street', 'city', 'state',
        'job', 'dob', 'trans_num', 'merchant'
    ]
    data.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    data['gender'] = data['gender'].map({'M': 0, 'F': 1})
    data = pd.get_dummies(data, columns=['category'], drop_first=True)

    X = data.drop(columns='is_fraud')
    y = data['is_fraud']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    result_text.pack_forget()
    messagebox.showinfo("Success", "Data preprocessed successfully!")

def display_data():
    global data
    if data is None:
        messagebox.showwarning("Warning", "Please load the dataset first!")
        return
    display = data.head()
    print("Displaying the first few rows:\n", display)

def train_logistic_regression():
    global models, X_train, y_train, X_test, y_test  # Add 'models' here
    if X_train is None or y_train is None:
        messagebox.showwarning("Warning", "Please preprocess the data first!")
        return

    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, "Training Logistic Regression, Please wait...")
    result_text.pack(padx=1, pady=1)
    result_frame.update()

    try:
        model = LogisticRegression(max_iter=1000, class_weight='balanced')
        model.fit(X_train, y_train)

        # Update the global 'models' dictionary
        models["Logistic Regression"] = model

        # Evaluate the model
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("\nROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        result_text.pack_forget()
        messagebox.showinfo("Success", "Logistic Regression model trained successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def train_random_forest():
    global models, X_train, y_train, X_test, y_test  # Add 'models' here
    if X_train is None or y_train is None:
        messagebox.showwarning("Warning", "Please preprocess the data first!")
        return

    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, "Training Random Forest, Please wait...")
    result_text.pack(padx=1, pady=1)
    result_frame.update()

    def train_model():
        try:
            print("Starting Random Forest model training...")
            model = RandomForestClassifier(
                n_estimators=50, max_depth=10, class_weight='balanced', random_state=42, verbose=1
            )
            model.fit(X_train, y_train)

            # Update the global 'models' dictionary
            models["Random Forest"] = model

            # Evaluate the model
            y_pred = model.predict(X_test)
            print("Accuracy:", accuracy_score(y_test, y_pred))

            result_text.pack_forget()
            messagebox.showinfo("Success", "Random Forest model trained successfully!")
        except Exception as e:
            print(f"Error during training: {e}")
            messagebox.showerror("Error", f"An error occurred: {e}")

    threading.Thread(target=train_model).start()

def train_xgboost():
    global models, X_train, y_train, X_test, y_test  # Add 'models' here
    if X_train is None or y_train is None:
        messagebox.showwarning("Warning", "Please preprocess the data first!")
        return

    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, "Training XGBoost, Please wait...")
    result_text.pack(padx=1, pady=1)
    result_frame.update()

    try:
        model = XGBClassifier(
            scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
            random_state=42, use_label_encoder=False, eval_metric='logloss'
        )
        model.fit(X_train, y_train)

        # Update the global 'models' dictionary
        models["XGBoost"] = model

        # Evaluate the model
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("\nROC AUC Score:")
        print(roc_auc_score(y_test, y_pred_proba))

        result_text.pack_forget()
        messagebox.showinfo("Success", "XGBoost model trained successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def train_isolation_forest():
    global models, X_train, X_test, y_train, y_test  # Add 'models' here
    if X_train is None or y_train is None:
        messagebox.showwarning("Warning", "Please preprocess the data first!")
        return

    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, "Training Isolation Forest, Please wait...")
    result_text.pack(padx=1, pady=1)
    result_frame.update()

    try:
        X_scaled = scaler.fit_transform(data.drop(columns='is_fraud'))
        X_scaled = pd.DataFrame(X_scaled, columns=data.drop(columns='is_fraud').columns)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, data['is_fraud'], test_size=0.2, random_state=42, stratify=data['is_fraud']
        )

        model = IsolationForest(contamination=0.01, random_state=42)
        model.fit(X_train)

        # Update the global 'models' dictionary
        models["Isolation Forest"] = model

        # Convert predictions: -1 (anomaly) -> 1 (fraud), 1 (normal) -> 0 (not fraud)
        y_pred = [1 if x == -1 else 0 for x in model.predict(X_test)]

        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        result_text.pack_forget()
        messagebox.showinfo("Success", "Isolation Forest model trained successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

def evaluate_model():
    global model
    if model is None or X_test is None or y_test is None:
        messagebox.showwarning("Warning", "Please train the model and prepare the test data first!")
        return

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, zero_division=0)
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, "Model Evaluation Report:\n")
    result_text.insert(tk.END, report)
    result_text.pack(padx=1, pady=1)

def clear_result_frame():
    for widget in result_frame.winfo_children():
        if widget != result_text:  # Preserve the result_text widget
            widget.destroy()

def evaluate_and_compare_models():
    global models, X_test, y_test
    if X_test is None or y_test is None:
        messagebox.showwarning("Warning", "Please preprocess the data and train the models first!")
        return

    # Evaluate each model and store ROC-AUC scores
    roc_auc_scores = {}
    plt.figure(figsize=(8, 6))

    for name, model in models.items():  # Iterate over all models
        if model is not None:  # Check if the model is trained
            try:
                if name == "Isolation Forest":
                    # Isolation Forest uses decision_function instead of predict_proba
                    y_pred_proba = [-x for x in model.decision_function(X_test)]
                else:
                    # Use predict_proba for other models
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Calculate ROC-AUC score
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                roc_auc_scores[name] = roc_auc

                # Plot ROC curve
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
            except Exception as e:
                print(f"Error evaluating {name}: {e}")

    # Check if any models were evaluated successfully
    if not roc_auc_scores:
        messagebox.showerror("Error", "No models were evaluated successfully. Please train models first.")
        return

    # Identify the best model
    best_model_name = max(roc_auc_scores, key=roc_auc_scores.get)
    best_score = roc_auc_scores[best_model_name]

    # Plot ROC-AUC curves
    plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-AUC Curves")
    plt.legend(loc="lower right")
    plt.show()

    messagebox.showinfo("Evaluation Results", f"Best Model: {best_model_name} (AUC = {best_score:.2f})")

# Function to predict using the best model
def predict_with_best_model():
    global models, X_test, y_test
    if X_test is None or y_test is None:
        messagebox.showwarning("Warning", "Please preprocess the data and train the models first!")
        return

    # Find the best model based on ROC-AUC score
    roc_auc_scores = {}
    for name, model in models.items():
        if model is not None:
            try:
                if name == "Isolation Forest":
                    y_pred_proba = [-x for x in model.decision_function(X_test)]
                else:
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                roc_auc_scores[name] = roc_auc
            except Exception as e:
                print(f"Error evaluating {name}: {e}")

    best_model_name = max(roc_auc_scores, key=roc_auc_scores.get)
    best_model = models[best_model_name]

    # Predict using the best model
    if best_model is not None:
        y_pred = best_model.predict(X_test)
        report = classification_report(y_test, y_pred, zero_division=0)
        result_text.delete(1.0, tk.END)
        result_text.insert(tk.END, f"Predictions using Best Model ({best_model_name}):\n")
        result_text.insert(tk.END, report)
        result_text.pack(padx=1, pady=1)
    else:
        messagebox.showwarning("Warning", "No trained models available!")

# New functions
def generate_sample_data():
    cc_number = credit_card_var.get().strip()
    if not cc_number:
        messagebox.showerror("Error", "Please enter a credit card number.")
        return pd.DataFrame(columns=['Date', 'Credit Card', 'Merchant', 'Category', 'Amount', 'City'])

    try:
        cc_number_int = int(cc_number)
    except ValueError:
        messagebox.showerror("Error", "Invalid credit card number. Please enter a valid number.")
        return pd.DataFrame(columns=['Date', 'Credit Card', 'Merchant', 'Category', 'Amount', 'City'])

    global data, filterData
    if data is None:
        messagebox.showerror("Error", "Dataset not loaded. Please load the dataset first.")
        return pd.DataFrame(columns=['Date', 'Credit Card', 'Merchant', 'Category', 'Amount', 'City'])

    # Verify if 'cc_num' column exists in the dataset
    if 'cc_num' not in data.columns:
        messagebox.showerror("Error", "The dataset does not contain the 'cc_num' column.")
        return pd.DataFrame(columns=['Date', 'Credit Card', 'Merchant', 'Category', 'Amount', 'City'])

    try:
        # Filter data based on the credit card number
        filterData = data[data["cc_num"] == cc_number_int]
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while filtering data: {e}")
        return pd.DataFrame(columns=['Date', 'Credit Card', 'Merchant', 'Category', 'Amount', 'City'])

    if filterData.empty:
        messagebox.showinfo("Info", f"No transactions found for credit card: {cc_number}")
        return pd.DataFrame(columns=['Date', 'Credit Card', 'Merchant', 'Category', 'Amount', 'City'])

    # Generate transaction data
    transactions = []
    for _, row in filterData.iterrows():
        transactions.append([
            row['trans_date_trans_time'],
            cc_number,
            row['merchant'],
            row['category'],
            row['amt'],
            row['city']
        ])
    return pd.DataFrame(transactions, columns=['Date', 'Credit Card', 'Merchant', 'Category', 'Amount', 'City'])

def track_expenses():
    global data, filterData  # Declare globals at the start
    try:
        # Hide the result_text widget if it's visible
        if result_text.winfo_ismapped():
            result_text.pack_forget()

        # Generate sample data for the entered credit card number
        data = generate_sample_data()

        # Exit early if no data is generated
        if data.empty:
            return

        # Update the UI with the new data
        update_ui()

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while tracking expenses: {e}")

def update_ui():
    global filterData

    # Clear existing rows in the treeview
    for row in tree.get_children():
        tree.delete(row)

    # Calculate total spent
    if filterData is None or filterData.empty:
        total_spent = 0
    else:
        total_spent = filterData['amt'].sum()

    limit = 1000  # Spending limit

    # Insert new rows into the treeview
    for _, row in data.iterrows():
        tree.insert("", "end", values=row.tolist())

    # Update total spent label and warning label
    total_spent_label.config(
        text=f"Total Spent: ${total_spent:.2f}",
        fg="red" if total_spent > limit else "green"
    )
    warning_label.config(
        text="⚠️ Overspending! Reduce Expenses!" if total_spent > limit else "✔ Spending is within limit.",
        fg="red" if total_spent > limit else "green"
    )

    # Plot the spending chart
    plot_expense_chart()

def plot_expense_chart():
    global data
    figure.clear()
    ax = figure.add_subplot(111)

    if not data.empty:
        category_totals = data.groupby('Category')['Amount'].sum()
        if not category_totals.empty:
            category_totals.plot(kind='bar', ax=ax, color=['#FF5733', '#33FF57', '#3357FF', '#FF33A1', '#FFC733'])
            ax.set_title("Spending by Category")
            ax.set_ylabel("Amount ($)")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            figure.tight_layout()
        else:
            ax.text(0.5, 0.5, "No Data Available", ha='center', va='center', fontsize=12)
    else:
        ax.text(0.5, 0.5, "No Data Available", ha='center', va='center', fontsize=12)

    canvas.draw()

def switch_to_expense_tracker():
    clear_result_frame()

    # Frame for centering content
    main_frame = tk.Frame(result_frame, bg="#f4f4f4")
    main_frame.place(relx=0.5, rely=0.5, anchor="center")

    title_label = tk.Label(main_frame, text="💰 Expense Tracker 💰", font=("Arial", 16, "bold"), bg="#f4f4f4", fg="#333")
    title_label.grid(row=0, column=0, columnspan=3, pady=(10, 20))

    # Input field
    tk.Label(main_frame, text="Credit Card Number:", bg="#f4f4f4", font=("Arial", 12)).grid(row=1, column=0, padx=10, pady=5)
    tk.Entry(main_frame, textvariable=credit_card_var, font=("Arial", 12), width=20).grid(row=1, column=1, padx=10, pady=5)
    tk.Button(main_frame, text="Track Expenses", font=("Arial", 12), bg="#008CBA", fg="white", command=track_expenses).grid(row=1, column=2, padx=10, pady=5)

    global total_spent_label
    total_spent_label = tk.Label(main_frame, text="Total Spent: $0.00", font=("Arial", 14, "bold"), bg="#f4f4f4")
    total_spent_label.grid(row=2, column=0, columnspan=3, pady=5)

    global warning_label
    warning_label = tk.Label(main_frame, text="✔ Spending is within limit.", font=("Arial", 12), fg="green", bg="#f4f4f4")
    warning_label.grid(row=3, column=0, columnspan=3, pady=5)

    columns = ['Date', 'Credit Card', 'Merchant', 'Category', 'Amount', 'City']
    tree_frame = tk.Frame(main_frame)
    tree_frame.grid(row=4, column=0, columnspan=3, pady=10)

    global tree
    tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=8)
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=120)
    
    scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side="right", fill="y")

    tree.pack()

    global figure, canvas
    figure = plt.Figure(figsize=(4, 3), dpi=100)
    canvas = FigureCanvasTkAgg(figure, master=main_frame)
    canvas.get_tk_widget().grid(row=5, column=0, columnspan=3, pady=10, sticky="nsew")

    main_frame.grid_rowconfigure(5, weight=1)
    main_frame.grid_columnconfigure(0, weight=1)

def clear_result_frame():
    for widget in result_frame.winfo_children():
        if widget != result_text:  # Preserve the result_text widget
            widget.destroy()

def show_spending_distribution():
    global data
    if data is None:
        messagebox.showwarning("Warning", "Please load the dataset first!")
        return

    clear_result_frame()

    spending_by_category = data.groupby('category')['amt'].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    spending_by_category.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title("Spending Distribution by Category")
    ax.set_xlabel("Category")
    ax.set_ylabel("Total Amount")
    plt.xticks(rotation=30)
    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=result_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)
    canvas.draw()

def show_transaction_pie_chart():
    global data
    if data is None:
        messagebox.showwarning("Warning", "Please load the dataset first!")
        return

    clear_result_frame()

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(data['category'].value_counts(), labels=data['category'].value_counts().index, autopct='%1.1f%%')
    ax.set_title("Transaction Categories")
    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=result_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)
    canvas.draw()

def show_fraud_proportion():
    global data
    if data is None:
        messagebox.showwarning("Warning", "Please load the dataset first!")
        return

    clear_result_frame()

    fraud_proportion = data['is_fraud'].value_counts(normalize=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(fraud_proportion, labels=['Non-Fraud', 'Fraud'], autopct='%1.1f%%', colors=['lightgreen', 'red'])
    ax.set_title("Proportion of Fraudulent vs Non-Fraudulent Transactions")
    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=result_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)
    canvas.draw()

def show_kde_plot():
    global data
    if data is None:
        messagebox.showwarning("Warning", "Please load the dataset first!")
        return

    clear_result_frame()

    # Ensure the dataset has the required columns
    if 'city_pop' not in data.columns or 'is_fraud' not in data.columns:
        messagebox.showwarning("Warning", "The dataset does not contain the required columns ('city_pop', 'is_fraud')!")
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.kdeplot(data=data, x='city_pop', hue='is_fraud', palette='coolwarm', fill=True, ax=ax)
    ax.set_title("KDE Plot: City Population by Fraud")
    ax.set_xlabel("City Population")
    ax.set_ylabel("Density")
    ax.legend(title='Is Fraud', labels=['Non-Fraud (0)', 'Fraud (1)'])
    plt.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=result_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)
    canvas.draw()

# Main application setup
root = tk.Tk()
root.title("Credit Card Fraud Detection")
root.geometry("900x600")

credit_card_var = tk.StringVar()

# Menu
menu_bar = tk.Menu(root)
root.config(menu=menu_bar)

file_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="File", menu=file_menu)
file_menu.add_command(label="Load Dataset", command=load_data)
file_menu.add_command(label="Display Data", command=display_data)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=root.quit)

model_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Model", menu=model_menu)
model_menu.add_command(label="Preprocess Data", command=preprocess_data)
model_menu.add_separator()
model_menu.add_command(label="Train Logistic Regression", command=train_logistic_regression)
model_menu.add_command(label="Train Random Forest", command=train_random_forest)
model_menu.add_command(label="Train XGBoost", command=train_xgboost)
model_menu.add_command(label="Train Isolation Forest", command=train_isolation_forest)
model_menu.add_separator()
model_menu.add_command(label="Evaluate All Models", command=evaluate_and_compare_models)
model_menu.add_command(label="Predict with Best Model", command=predict_with_best_model)

others_menu = tk.Menu(menu_bar, tearoff=0)
menu_bar.add_cascade(label="Other", menu=others_menu)
others_menu.add_command(label="Spending Distribution by Category", command=show_spending_distribution)
others_menu.add_command(label="Transaction Categories Pie Chart", command=show_transaction_pie_chart)
others_menu.add_command(label="Fraud Proportion", command=show_fraud_proportion)
others_menu.add_command(label="KDE Plot: City Population by Fraud", command=show_kde_plot)
others_menu.add_command(label="Switch to Expense Tracker", command=switch_to_expense_tracker)

# Results Frame
result_frame = ttk.Frame(root)
result_frame.pack(pady=1, fill=tk.BOTH, expand=True)

result_text = tk.Text(result_frame, height=10, width=100)
result_text.pack_forget()

root.mainloop()