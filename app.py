from flask import Flask, render_template
import random

app = Flask(__name__)

TOTAL_SLOTS = 5

@app.route('/')
def home():

    parking_slots = []

    occupied_count = 0

    for _ in range(TOTAL_SLOTS):

        slot = random.choice([0,1])

        parking_slots.append(slot)

        if slot == 1:
            occupied_count += 1

    free_count = TOTAL_SLOTS - occupied_count

    vehicle_queue = random.randint(1,10)

    return render_template(
        'index.html',
        parking_slots=parking_slots,
        occupied_count=occupied_count,
        free_count=free_count,
        vehicle_queue=vehicle_queue
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5002)