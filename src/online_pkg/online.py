import numpy as np
import serial
import time
import matplotlib.pyplot as plt

# Battery parameters
class BatteryParam:
    def __init__(self):
        self.R0 = 0.072  # Ω
        self.Rp = 0.03   # Ω
        self.Cp = 1000   # F
        self.Qn = 1.5 * 3600.0  # Capacity (As)
        self.deltaT = 1   # Sampling period (s)
        self.eta = 1.0    # Coulomb efficiency

# Coefficients matrix for state-space model (linear)
class CoeffMatrix:
    def __init__(self):
        self.A = np.zeros((2, 2))
        self.B = np.zeros(2)
        self.D = 0

# Battery data
class Battery:
    def __init__(self, Cur=0, Vol=0):
        self.Cur = Cur  # Current data
        self.Vol = Vol  # Voltage data

# Noise parameters
class NoiseParam:
    def __init__(self):
        self.ww = 0.0001
        self.vv = 0.001
        self.Gw = np.identity(2) * 1e-3
        self.Gv = 0.01

# EKF Gain parameters
class EKFGainParam:
    def __init__(self):
        self.Q = np.identity(2) * 1e-8
        self.R = 0.01
        self.P0 = np.zeros((2, 2))
        self.P0[0, 0] = 0.1
        self.P0[1, 1] = 1.0

# Initial state parameters
class InitParam:
    def __init__(self):
        self.x1 = np.array([0.01, 0.8])
        self.x1_ = np.array([0.012, 0.7])
        self.BG1_diag = np.zeros((2, 2))
        self.BG1_diag[0, 0] = 0.03
        self.BG1_diag[1, 1] = 0.01

# Estimate Output
class EstimateOutput:
    def __init__(self):
        self.x_hat = np.zeros(2)
        self.x_low = np.zeros(2)
        self.x_up = np.zeros(2)
        self.y_hat = 0
        self.y_low = 0
        self.y_up = 0

# Simulation system for true state and output
class SimSystem:
    def __init__(self):
        self.x = np.zeros(2)
        self.y = 0

# Dimension parameters
class Dimension:
    def __init__(self, s, n):
        self.s = s
        self.n = n

# Coefficients matrix creation based on battery parameters
def makeParaMatrix(bp):
    matrix = CoeffMatrix()
    e = np.exp(-bp.deltaT / (bp.Rp * bp.Cp))
    matrix.A[0, 0] = e
    matrix.A[1, 1] = 1
    matrix.B[0] = bp.Rp * (1 - e)
    matrix.B[1] = -bp.eta * bp.deltaT / bp.Qn
    matrix.D = -bp.R0
    return matrix

# Read serial data (for this example we use pyserial)
def read_serial_data(serial_port, baudrate=115200):
    try:
        ser = serial.Serial(serial_port, baudrate)
        return ser
    except Exception as e:
        print(f"Error opening serial port: {e}")
        return None

# Parse data
def parse_serial_data(data):
    print(f"Raw data: {data}")  # Debugging the raw data
    try:
        values = data.split(",")
        if len(values) < 3:
            print("Error: Incorrect data format")
            return None
        U = float(values[0]) / 1000.0
        U_p = float(values[1]) / 1000.0
        Rp = U
        I_k = U_p / Rp
        return U, I_k
    except Exception as e:  
        print(f"Error parsing data: {e}")
        return None


# EKF (Extended Kalman Filter) implementation
def EKF_offline(cm, dim, pfit, input, noise, gain, output, BG_current):
    x_pre = np.dot(cm.A, output.x_hat) + np.dot(cm.B, input.Cur)
    BG_pre = np.dot(cm.A, BG_current)
    C = np.array([-1, polyval(polyder(pfit), x_pre[1])])
    Uoc = polyval(pfit, x_pre[1])
    Up = x_pre[0]
    output.y_hat = Uoc - Up + gain.R * input.Cur
    BG_y = np.dot(C, BG_current)
    BG_y = jiangjie(BG_y, dim.s)
    output.y_low = output.y_hat - np.abs(BG_y).sum()
    output.y_up = output.y_hat + np.abs(BG_y).sum()

    P_pred = np.dot(cm.A, np.dot(gain.P0, cm.A.T)) + gain.Q
    S = np.dot(C, np.dot(P_pred, C.T)) + gain.R
    K = np.dot(P_pred, C.T) * (1.0 / S)

    gain.P0 = np.dot(np.identity(output.x_hat.size) - np.dot(K, C), P_pred)
    x_updated = x_pre + np.dot(K, (input.Vol - output.y_hat))
    output.x_hat = x_updated
    BG_current = jiangjie(BG_current, dim.s)

    abs_sum = np.abs(BG_current).sum(axis=1)
    output.x_low = x_updated - abs_sum
    output.x_up = x_updated + abs_sum

# Function to compute polynomial value
def polyval(coeffs, x):
    return np.sum([c * x**(len(coeffs) - 1 - i) for i, c in enumerate(coeffs)])

# Function to compute polynomial derivative coefficients
def polyder(coeffs):
    return np.array([coeffs[i] * (len(coeffs) - 1 - i) for i in range(len(coeffs) - 1)])

# Function to generate boundary reduction
def jiangjie(G, s):
    r = G.shape[1]
    G_sorted = sorted(range(r), key=lambda i: np.linalg.norm(G[:, i]), reverse=True)
    G_sorted = np.array(G_sorted)
    G_sorted = G[:, G_sorted]
    return G_sorted

# Plot results
def plot_results(k_values, x_hat, x_low, x_up, y_hat, y_low, y_up):
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, x_hat, label='x_hat')
    plt.plot(k_values, x_low, label='x_low')
    plt.plot(k_values, x_up, label='x_up')
    plt.plot(k_values, y_hat, label='y_hat')
    plt.plot(k_values, y_low, label='y_low')
    plt.plot(k_values, y_up, label='y_up')
    plt.xlabel('Time (k)')
    plt.ylabel('Values')
    plt.legend()
    plt.title('State and Output Estimations')
    plt.grid()
    plt.show()

# Main function
def main():
    bp = BatteryParam()
    cm = makeParaMatrix(bp)
    pfit = np.loadtxt("../test/pfit.txt")  # Load the SOC nonlinear observation coefficients
    dim = Dimension(12, 2)
    init = InitParam()
    noise = NoiseParam()
    gain = EKFGainParam()

    serial_port = "/dev/ttyUSB0"
    serial_fd = read_serial_data(serial_port)
    if serial_fd is None:
        return

    k_values = []
    x_hat_vals = []
    x_low_vals = []
    x_up_vals = []
    y_hat_vals = []
    y_low_vals = []
    y_up_vals = []

    k = 0
    while True:
        line = serial_fd.readline().decode('utf-8').strip()
        if line:
            U, I_k = parse_serial_data(line)
            if U is not None:
                input = Battery(-I_k, U)
                sim = SimSystem()
                sim.x = init.x1
                output = EstimateOutput()
                output.x_hat = init.x1_
                output.x_low = init.x1_ - init.BG1_diag.sum(axis=0)
                output.x_up = init.x1_ + init.BG1_diag.sum(axis=0)
                BG_current = init.BG1_diag
                EKF_offline(cm, dim, pfit, input, noise, gain, output, BG_current)

                k_values.append(k)
                x_hat_vals.append(output.x_hat[0])
                x_low_vals.append(output.x_low[0])
                x_up_vals.append(output.x_up[0])
                y_hat_vals.append(output.y_hat)
                y_low_vals.append(output.y_low)
                y_up_vals.append(output.y_up)

                print(f"k={k}, U={U}, I={I_k}")
                k += 1

        time.sleep(1)

    plot_results(k_values, x_hat_vals, x_low_vals, x_up_vals, y_hat_vals, y_low_vals, y_up_vals)

if __name__ == "__main__":
    main()
