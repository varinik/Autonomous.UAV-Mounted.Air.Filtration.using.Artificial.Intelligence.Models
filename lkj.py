import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# NumPy Gaussian Plume for data generation
def gaussian_plume_np(x, y, z, xs, ys, H, Q, u=1.0):
    sigma_y = 0.08 * x * (1 + 0.0001 * x) ** (-0.5)
    sigma_z = 0.06 * x * (1 + 0.0015 * x) ** (-0.5)
    term1 = Q / (2 * np.pi * u * sigma_y * sigma_z)
    term2 = np.exp(-(y - ys) ** 2 / (2 * sigma_y ** 2))
    term3 = np.exp(-(z - H) ** 2 / (2 * sigma_z ** 2)) + np.exp(-(z + H) ** 2 / (2 * sigma_z ** 2))
    return term1 * term2 * term3

# TensorFlow version for physics loss
def gaussian_plume(x, y, z, xs, ys, H, Q, u=1.0):
    sigma_y = 0.08 * x * tf.pow(1 + 0.0001 * x, -0.5)
    sigma_z = 0.06 * x * tf.pow(1 + 0.0015 * x, -0.5)
    term1 = Q / (2 * tf.constant(np.pi, dtype=tf.float32) * u * sigma_y * sigma_z)
    term2 = tf.exp(-tf.square(y - ys) / (2 * tf.square(sigma_y)))
    term3 = tf.exp(-tf.square(z - H) / (2 * tf.square(sigma_z))) + tf.exp(-tf.square(z + H) / (2 * tf.square(sigma_z)))
    return term1 * term2 * term3

# Data generation (NumPy)
def generate_data(n_samples=100):
    X, Y = [], []
    for _ in range(n_samples):
        xs = np.random.uniform(0, 5)
        ys = np.random.uniform(0, 5)
        zs = np.random.uniform(0, 5)
        H = 0
        Q = 100
        cur_time = []
        for i in range(100):
            points = []
            for j in range(3):
                x = xs - 0.0001*i + j*0.25
                y = ys - 0.0001*i + j*0.25
                z = zs - 0.0001*i + j*0.25
                points.append([x, y, z])
            points = np.array(points)
            concentrations = np.array([gaussian_plume_np(points[i, 0], points[i, 1], points[i, 2], xs, ys, H, Q) for i in range(3)])
            concentrations = concentrations.reshape(-1, 1)
            combined = np.hstack((points, concentrations))
            cur_time.append(combined)
        X.append(cur_time)
        Y.append([xs, ys, zs])
    return np.array(X), np.array(Y)

# Physics-informed loss function (TensorFlow)
def physics_informed_loss(X_batch, y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    physics_loss = 0.0
    for i in range(3):
        x_i = X_batch[:, i, 0]
        y_i = X_batch[:, i, 1]
        z_i = X_batch[:, i, 2]
        C_i = X_batch[:, i, 3]

        xs_pred = y_pred[:, 0]
        ys_pred = y_pred[:, 1]
        H_pred = y_pred[:, 2]
        Q_pred = y_pred[:, 3]

        C_pred = gaussian_plume(x_i, y_i, z_i, xs_pred, ys_pred, H_pred, Q_pred)
        print(C_pred)
        # if(C_pred<1):
        #     C_pred = 1
        physics_loss += tf.reduce_mean(tf.square(C_i - C_pred))

    return mse_loss + 0.1 * physics_loss

# Generate and normalize data
n_samples = 10000
X, y = generate_data(n_samples = 1)
scaler_coords = MinMaxScaler()
scaler_conc = MinMaxScaler()
scaler_y = MinMaxScaler()

X_coords = scaler_coords.fit_transform(X[:, :, :3].reshape(-1, 3)).reshape(X.shape[0], 3, 3)
X_conc = scaler_conc.fit_transform(X[:, :, 3:4].reshape(-1, 1)).reshape(X.shape[0], 3, 1)
X_normalized = np.concatenate([X_coords, X_conc], axis=-1)
y_normalized = scaler_y.fit_transform(y)

# Convert to tensors
X_tensor = tf.convert_to_tensor(X_normalized, dtype=tf.float32)
y_tensor = tf.convert_to_tensor(y_normalized, dtype=tf.float32)

# Build model
model = Sequential([
    LSTM(64, input_shape=(3, 4), return_sequences=True),
    LSTM(32),
    Dense(16, activation='relu'),
    Dense(4)
])
optimizer = tf.keras.optimizers.Adam()

# Custom training loop
batch_size = 32
epochs = 50
dataset = tf.data.Dataset.from_tensor_slices((X_tensor, y_tensor)).shuffle(1000).batch(batch_size)

for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    for step, (X_batch, y_batch) in enumerate(dataset):
        with tf.GradientTape() as tape:
            y_pred = model(X_batch, training=True)
            loss = physics_informed_loss(X_batch, y_batch, y_pred)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # if step % 100 == 0:
        #     print(f"Step {step}, Loss: {loss.numpy():.4f}")

# Test prediction
test_points = np.array([[100, 50, 0], [200, -50, 0], [0, 0, 0]])
test_conc = [gaussian_plume_np(test_points[i, 0], test_points[i, 1], test_points[i, 2], xs=150, ys=0, H=50, Q=500)
             for i in range(3)]
test_input = np.concatenate([test_points, np.array(test_conc)[:, None]], axis=1)
test_coords = scaler_coords.transform(test_input[:, :3])
test_conc_scaled = scaler_conc.transform(test_input[:, 3:4])
test_input_normalized = np.concatenate([test_coords, test_conc_scaled], axis=-1).reshape(1, 3, 4)

pred_normalized = model.predict(test_input_normalized)
pred = scaler_y.inverse_transform(pred_normalized)
print("Predicted source [xs, ys, H, Q]:", pred)


# # import numpy as np
# # import tensorflow as tf
# # from tensorflow.keras.models import Sequential
# # from tensorflow.keras.layers import LSTM, Dense
# # from sklearn.preprocessing import MinMaxScaler

# # # Gaussian Plume Model
# # def gaussian_plume(x, y, z, xs, ys, H, Q, u=1.0, stability='B'):
# #     print(type(x), type(y), type(z), type(ys), type(H), type(Q))
# #     sigma_y = 0.08 * x * (1 + 0.0001 * x) ** (-0.5)
# #     sigma_z = 0.06 * x * (1 + 0.0015 * x) ** (-0.5)
# #     term1 = Q / (2 * np.pi * u * sigma_y * sigma_z)
# #     term2 = np.exp(-(y - ys)**2 / (2 * sigma_y**2))
# #     term3 = np.exp(-(z - H)**2 / (2 * sigma_z**2)) + np.exp(-(z + H)**2 / (2 * sigma_z**2))
# #     return term1 * term2 * term3

# # # Generate Data
# # def generate_data(n_samples, x_range=(-1000, 1000), y_range=(-1000, 1000), H_range=(10, 100), Q_range=(100, 1000)):
# #     X = []
# #     y = []
# #     for _ in range(n_samples):
# #         xs = np.random.uniform(0, 50)
# #         ys = np.random.uniform(0, 50)
# #         print(type(ys))
# #         H = 0   # This will always return 0
# #         Q = 100
# #         points = np.random.uniform(0, 50, (3, 2))
# #         z = np.zeros(3)
# #         concentrations = [gaussian_plume(points[i, 0], points[i, 1], z[i], xs, ys, H, Q) * np.random.normal(1, 0.05) for i in range(3)]
# #         input_data = np.concatenate([points, z[:, None], np.array(concentrations)[:, None]], axis=1)
# #         X.append(input_data)
# #         y.append([xs, ys, H, Q])
# #     return np.array(X), np.array(y)

# # # Physics-Informed Loss
# # def physics_informed_loss(y_true, y_pred, X):
# #     mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
# #     physics_loss = 0
# #     for i in range(3):
# #         x_i, y_i, z_i, C_i = X[:, i, 0], X[:, i, 1], X[:, i, 2], X[:, i, 3]
# #         xs_pred, ys_pred, H_pred, Q_pred = y_pred[:, 0], y_pred[:, 1], y_pred[:, 2], y_pred[:, 3]
# #         C_pred = gaussian_plume(x_i, y_i, z_i, xs_pred, ys_pred, H_pred, Q_pred)
# #         physics_loss += tf.reduce_mean(tf.square(C_i - C_pred))
# #     return mse_loss + 0.1 * physics_loss

# # # Prepare Data
# # n_samples = 10000
# # X, y = generate_data(n_samples)
# # scaler_coords = MinMaxScaler()
# # scaler_concentrations = MinMaxScaler()
# # scaler_y = MinMaxScaler()
# # X_coords = scaler_coords.fit_transform(X[:, :, :3].reshape(-1, 3)).reshape(X.shape[0], 3, 3)
# # X_concentrations = scaler_concentrations.fit_transform(X[:, :, 3:4].reshape(-1, 1)).reshape(X.shape[0], 3, 1)
# # X_normalized = np.concatenate([X_coords, X_concentrations], axis=-1)
# # y_normalized = scaler_y.fit_transform(y)

# # print("here")

# # # Build LSTM Model
# # model = Sequential([
# #     LSTM(64, input_shape=(3, 4), return_sequences=True),
# #     LSTM(32),
# #     Dense(16, activation='relu'),
# #     Dense(4)
# # ])

# # # Compile and Train
# # model.compile(optimizer='adam', loss=lambda y_true, y_pred: physics_informed_loss(y_true, y_pred, X_normalized))
# # model.fit(X_normalized, y_normalized, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# # # Test Prediction
# # test_points = np.array([[100, 50, 0], [200, -50, 0], [0, 0, 0]])
# # test_concentrations = [gaussian_plume(test_points[i, 0], test_points[i, 1], test_points[i, 2], xs=150, ys=0, H=50, Q=500) for i in range(3)]
# # test_input = np.concatenate([test_points, np.array(test_concentrations)[:, None]], axis=1)
# # test_coords = scaler_coords.transform(test_input[:, :3])
# # test_conc = scaler_concentrations.transform(test_input[:, 3:4])
# # test_input_normalized = np.concatenate([test_coords, test_conc], axis=-1).reshape(1, 3, 4)
# # pred_normalized = model.predict(test_input_normalized)
# # pred = scaler_y.inverse_transform(pred_normalized)
# # print("Predicted source [xs, ys, H, Q]:", pred)

# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from sklearn.preprocessing import MinMaxScaler

# # TensorFlow-compatible Gaussian Plume function
# def gaussian_plume(x, y, z, xs, ys, H, Q, u=1.0):
#     # Use tf operations instead of np
#     sigma_y = 0.08 * x * tf.pow(1 + 0.0001 * x, -0.5)
#     sigma_z = 0.06 * x * tf.pow(1 + 0.0015 * x, -0.5)
#     term1 = Q / (2 * tf.constant(np.pi, dtype=tf.float32) * u * sigma_y * sigma_z)
#     term2 = tf.exp(-tf.square(y - ys) / (2 * tf.square(sigma_y)))
#     term3 = tf.exp(-tf.square(z - H) / (2 * tf.square(sigma_z))) + tf.exp(-tf.square(z + H) / (2 * tf.square(sigma_z)))
#     return term1 * term2 * term3

# # NumPy-only data generation function (no change needed)
# def generate_data(n_samples, x_range=(0, 50), y_range=(0, 50)):
#     X = []
#     y = []
#     for _ in range(n_samples):
#         xs = np.random.uniform(*x_range)
#         ys = np.random.uniform(*y_range)
#         H = 0
#         Q = 100
#         points = np.random.uniform(0, 50, (3, 2))
#         z = np.zeros(3)
#         concentrations = [gaussian_plume_np(points[i, 0], points[i, 1], z[i], xs, ys, H, Q) * np.random.normal(1, 0.05) for i in range(3)]
#         input_data = np.concatenate([points, z[:, None], np.array(concentrations)[:, None]], axis=1)
#         X.append(input_data)
#         y.append([xs, ys, H, Q])
#     return np.array(X), np.array(y)

# # Original NumPy gaussian plume used only in data generation
# def gaussian_plume_np(x, y, z, xs, ys, H, Q, u=1.0):
#     sigma_y = 0.08 * x * (1 + 0.0001 * x) ** (-0.5)
#     sigma_z = 0.06 * x * (1 + 0.0015 * x) ** (-0.5)
#     term1 = Q / (2 * np.pi * u * sigma_y * sigma_z)
#     term2 = np.exp(-(y - ys)**2 / (2 * sigma_y**2))
#     term3 = np.exp(-(z - H)**2 / (2 * sigma_z**2)) + np.exp(-(z + H)**2 / (2 * sigma_z**2))
#     return term1 * term2 * term3

# # Physics-informed loss adapted for TensorFlow tensors
# def physics_informed_loss(X):
#     def loss(y_true, y_pred):
#         mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
#         physics_loss = 0
#         for i in range(3):
#             x_i = X[:, i, 0]
#             y_i = X[:, i, 1]
#             z_i = X[:, i, 2]
#             C_i = X[:, i, 3]

#             xs_pred = y_pred[:, 0]
#             ys_pred = y_pred[:, 1]
#             H_pred  = y_pred[:, 2]
#             Q_pred  = y_pred[:, 3]

#             C_pred = gaussian_plume(x_i, y_i, z_i, xs_pred, ys_pred, H_pred, Q_pred)
#             physics_loss += tf.reduce_mean(tf.square(C_i - C_pred))

#         return mse_loss + 0.1 * physics_loss
#     return loss

# # Prepare data
# n_samples = 10000
# X, y = generate_data(n_samples)
# scaler_coords = MinMaxScaler()
# scaler_concentrations = MinMaxScaler()
# scaler_y = MinMaxScaler()
# X_coords = scaler_coords.fit_transform(X[:, :, :3].reshape(-1, 3)).reshape(X.shape[0], 3, 3)
# X_concentrations = scaler_concentrations.fit_transform(X[:, :, 3:4].reshape(-1, 1)).reshape(X.shape[0], 3, 1)
# X_normalized = np.concatenate([X_coords, X_concentrations], axis=-1)
# y_normalized = scaler_y.fit_transform(y)

# # Build LSTM model
# model = Sequential([
#     LSTM(64, input_shape=(3, 4), return_sequences=True),
#     LSTM(32),
#     Dense(16, activation='relu'),
#     Dense(4)
# ])

# # Compile model with physics-informed loss
# model.compile(optimizer='adam', loss=physics_informed_loss(tf.constant(X_normalized, dtype=tf.float32)))

# # Train
# model.fit(X_normalized, y_normalized, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# # Test prediction
# test_points = np.array([[100, 50, 0], [200, -50, 0], [0, 0, 0]])
# test_concentrations = [gaussian_plume_np(test_points[i, 0], test_points[i, 1], test_points[i, 2], xs=150, ys=0, H=50, Q=500) for i in range(3)]
# test_input = np.concatenate([test_points, np.array(test_concentrations)[:, None]], axis=1)
# test_coords = scaler_coords.transform(test_input[:, :3])
# test_conc = scaler_concentrations.transform(test_input[:, 3:4])
# test_input_normalized = np.concatenate([test_coords, test_conc], axis=-1).reshape(1, 3, 4)
# pred_normalized = model.predict(test_input_normalized)
# pred = scaler_y.inverse_transform(pred_normalized)
# print("Predicted source [xs, ys, H, Q]:", pred)
