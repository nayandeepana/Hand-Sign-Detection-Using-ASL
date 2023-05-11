# import os
# import numpy as np
# from sklearn.neighbors import KNeighborsClassifier

# # Load the data
# X = []
# y = []
# for filename in os.listdir('data'):
#     if filename.endswith('.npy'):
#         data = np.load(f"data/{filename}")
#         X.append(data)
#         label = filename.split('_')[0]
#         y.append(label)

# # Convert the data to numpy arrays
# X = np.array(X)
# y = np.array(y)

# # Split the data into training and testing sets
# split_index = int(len(X) * 0.8)
# X_train = X[:split_index]
# y_train = y[:split_index]
# X_test = X[split_index:]
# y_test = y[split_index:]

# # Train the KNN model
# model = KNeighborsClassifier(n_neighbors=3)
# model.fit(X_train, y_train)

# # Evaluate the model
# accuracy = model.score(X_test, y_test)
# print(f"Accuracy: {accuracy:.2f}")

# # Save the model
# if not os.path.exists('models'):
#     os.makedirs('models')
# np.save('models/knn_model.npy', model)

import os
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Load the data
data = []
labels = []
for filename in os.listdir('data'):
    if filename.endswith('.npy'):
        label = filename.split('_')[0]
        labels.append(label)
        data.append(np.load(os.path.join('data', filename)))

# Convert the data and labels to numpy arrays
data = np.array(data)
labels = np.array(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train the KNN model
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(data, labels)

# Evaluate the accuracy of the model on the testing set
accuracy = knn.score(X_test, y_test)
print(f'Accuracy: {accuracy:.2f}')

# Save the model
if not os.path.exists('models'):
    os.makedirs('models')
filename = f'models/knn_model_k{k}.joblib'
joblib.dump(knn, filename)
