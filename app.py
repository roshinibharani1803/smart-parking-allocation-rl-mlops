from flask import Flask, render_template, redirect, url_for
import numpy as np
import glob
import os

from env.parking_env import ParkingEnv

app = Flask(__name__)

# ==========================
# Load Environment
# ==========================
env = ParkingEnv()

# ==========================
# Load Latest Trained Model
# ==========================
model_files = glob.glob("models/q_table_*.npy")

if not model_files:
    raise FileNotFoundError(
        "No trained Q-table found! Train the model first."
    )

latest_model = max(
    model_files,
    key=os.path.getctime
)

q_table = np.load(latest_model)

print(f"\nLoaded Model: {latest_model}")

# ==========================
# Initial Parking State
# ==========================
current_state = np.array(
    np.random.choice(
        [0, 1],
        size=env.total_slots,
        p=[0.6, 0.4]
    )
)

# Dashboard variables
total_vehicles = 0
last_reward = 0
last_selected_slot = None
allocation_status = "Waiting for incoming vehicle..."
vehicle_queue = np.random.randint(2, 5)


# ==========================
# HOME PAGE
# ==========================
@app.route("/")
def home():

    global current_state
    global total_vehicles
    global last_reward
    global last_selected_slot
    global allocation_status
    global vehicle_queue

    slots = []

    free_slots = 0
    occupied_slots = 0

    selected_slot = None

    if np.any(current_state == 0):

        state_index = int(
            "".join(map(str, current_state)),
            2
        )

        selected_slot = int(
            np.argmax(q_table[state_index])
        )

    for i, value in enumerate(current_state):

        slot = {
            "id": f"P{i+1}",
            "status": "",
            "distance": env.slot_info[i]["distance"],
            "charger": env.slot_info[i]["charger"],
            "vip": env.slot_info[i]["vip"]
        }

        if value == 0:

            free_slots += 1

            if i == selected_slot:
                slot["status"] = "selected"
            else:
                slot["status"] = "free"

        else:

            occupied_slots += 1
            slot["status"] = "occupied"

        slots.append(slot)

    occupancy = int(
        (occupied_slots / env.total_slots) * 100
    )

    return render_template(
        "index.html",
        slots=slots,
        occupancy=occupancy,
        free_slots=free_slots,
        occupied_slots=occupied_slots,
        total_vehicles=total_vehicles,
        last_reward=last_reward,
        last_selected_slot=last_selected_slot,
        allocation_status=allocation_status,
        parking_state=current_state.tolist(),
        vehicle_queue=vehicle_queue
    )


# ==========================
# ALLOCATE VEHICLE
# ==========================
@app.route("/allocate")
def allocate():

    global current_state
    global total_vehicles
    global last_reward
    global last_selected_slot
    global allocation_status
    global vehicle_queue

    if np.all(current_state == 1):

        allocation_status = "Parking lot is full!"

        return redirect(url_for("home"))

    state_index = int(
        "".join(map(str, current_state)),
        2
    )

    # Always exploit learned policy
    action = int(
        np.argmax(q_table[state_index])
    )

    last_selected_slot = f"P{action+1}"

    # Use actual environment
    env.state = current_state.copy()

    next_state, reward, done, _, _ = env.step(action)

    current_state = next_state

    last_reward = reward

    if reward >= 0:

        allocation_status = (
            f"Vehicle allocated successfully to P{action+1}"
        )

        total_vehicles += 1

        vehicle_queue = max(
            0,
            vehicle_queue - 1
        )

        if np.random.rand() < 0.5:
            vehicle_queue += 1

    else:

        allocation_status = (
            f"Allocation failed! P{action+1} is already occupied."
        )

    return redirect(url_for("home"))


# ==========================
# RESET ENVIRONMENT
# ==========================
@app.route("/reset")
def reset():

    global current_state
    global total_vehicles
    global last_reward
    global last_selected_slot
    global allocation_status
    global vehicle_queue

    current_state = np.array(
        np.random.choice(
            [0, 1],
            size=env.total_slots,
            p=[0.6, 0.4]
        )
    )

    total_vehicles = 0

    last_reward = 0

    last_selected_slot = None

    vehicle_queue = np.random.randint(2, 5)

    allocation_status = (
        "Parking environment reset successfully."
    )

    return redirect(url_for("home"))


# ==========================
# RUN APP
# ==========================
if __name__ == "__main__":

    app.run(debug=True)