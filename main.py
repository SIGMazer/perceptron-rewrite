import os
import random
import numpy as np
from math import floor
import time

HEIGHT = 64
WIDTH = 64
PPM_SCALER = 4
PPM_RANGE = 1.0
PPM_COLOR_INTENSITY = 255
DATA_FOLDER = "./data"
SAMPLE_SIZE = 1000
BIAS = 0.5
TRAIN_PASSES = 100
INT_MAX = 2 ** 31 - 1

def clamp(x, low, high):
    return max(low, min(x, high))

def fill_rect(layer, x, y, w, h, value):
    x0 = clamp(x, 0, WIDTH-1)
    y0 = clamp(y, 0, HEIGHT-1)
    x1 = clamp(x0 + w - 1, 0, WIDTH-1)
    y1 = clamp(y0 + h - 1, 0, HEIGHT-1)
    layer[y0:y1+1, x0:x1+1] = value

def fill_circle(layer, cx, cy, r, value):
    x0 = clamp(cx - r, 0, WIDTH-1)
    y0 = clamp(cy - r, 0, HEIGHT-1)
    x1 = clamp(cx + r, 0, WIDTH-1)
    y1 = clamp(cy + r, 0, HEIGHT-1)

    y, x = np.ogrid[y0:y1+1, x0:x1+1]
    mask = ((x - cx) ** 2 + (y - cy) ** 2) <= r ** 2
    layer[y0:y1+1, x0:x1+1][mask] = value

def save_as_ppm(layer, file_path):
    ppm_data = np.zeros((HEIGHT * PPM_SCALER, WIDTH * PPM_SCALER, 3), dtype=np.uint8)

    for y in range(HEIGHT * PPM_SCALER):
        for x in range(WIDTH * PPM_SCALER):
            s = (layer[y // PPM_SCALER, x // PPM_SCALER] + PPM_RANGE) / (2.0 * PPM_RANGE)
            pixel = (
                floor(PPM_COLOR_INTENSITY * (1.0 - s)),
                floor(PPM_COLOR_INTENSITY * (1.0 - s)),
                floor(PPM_COLOR_INTENSITY * 0)
            )
            ppm_data[y, x] = pixel

    with open(file_path, 'wb') as f:
        line = f'P6\n{WIDTH * PPM_SCALER} {HEIGHT * PPM_SCALER}\n255\n'
        f.write(line.encode())
        f.write(ppm_data.tobytes())

def save_as_bin(layer, file_path):
    layer.tofile(file_path)


def predict(inputs, weights):
    return np.sum(inputs * weights)

def add_to_weights(inputs, weights):
    weights += inputs

def sub_to_weights(inputs, weights):
    weights -= inputs

def rand_range(low, high):
    if low >= high:
        return low
    return random.randint(low, high -1) 

def random_rect(layer):
    layer[:, :] = 0.0
    x = rand_range(0, WIDTH)
    y = rand_range(0, HEIGHT)
    w = rand_range(1, WIDTH - x + 1)
    h = rand_range(1, HEIGHT - y + 1)
    fill_rect(layer, x, y, w, h, 1.0)

def random_circle(layer):
    layer[:, :] = 0.0
    cx = rand_range(0, WIDTH)
    cy = rand_range(0, HEIGHT)
    r = rand_range(1, min(cx, cy, WIDTH - cx, HEIGHT - cy, INT_MAX))
    fill_circle(layer, cx, cy, r, 1.0)

def train_pass(inputs, weights, count):
    adjusted = 0

    for _ in range(SAMPLE_SIZE):
        random_rect(inputs)
        if predict(inputs, weights) > BIAS:
            sub_to_weights(inputs, weights)
            file_path = os.path.join(DATA_FOLDER, f'weights-{count:03d}.ppm')
            print(f"[INFO] saving {file_path}")
            save_as_ppm(weights, file_path)
            adjusted += 1
            count += 1

        random_circle(inputs)
        if predict(inputs, weights) < BIAS:
            add_to_weights(inputs, weights)
            file_path = os.path.join(DATA_FOLDER, f'weights-{count:03d}.ppm')
            print(f"[INFO] saving {file_path}")
            save_as_ppm(weights, file_path)
            adjusted += 1
            count += 1

    return adjusted

def check_pass(inputs, weights):
    adjusted = 0

    for _ in range(SAMPLE_SIZE):
        random_rect(inputs)
        if predict(inputs, weights) > BIAS:
            adjusted += 1

        random_circle(inputs)
        if predict(inputs, weights) < BIAS:
            adjusted += 1

    return adjusted

if __name__ == "__main__":
    print("[INFO] creating", DATA_FOLDER)
    os.makedirs(DATA_FOLDER, exist_ok=True)

    inputs = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    weights = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    adj = 0;
    count = 0

    random.seed(int(time.time()))
    for i in range(TRAIN_PASSES):
        adj = train_pass(inputs, weights, count)
        print(f"[INFO] Pass {i}: adjusted {adj} times")
        if adj <= 0:
            break

    random.seed(42)  # CHECK_SEED
    adj = check_pass(inputs, weights)
    print(f"[INFO] fail rate of trained model is {adj / (SAMPLE_SIZE * 2.0)}")

