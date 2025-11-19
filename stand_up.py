# stand_up.py
from control import Control
import time

def stand_up():
    bot = Control()

    print("Moving to neutral pose...")
    bot.move_position(0, 0, 0)
    time.sleep(1)

    print("Starting stand-up sequence...")

    start_z = 0       # neutral height
    end_z   = -80     # how high body lifts (adjust between -60 and -100)
    steps   = 20

    for i in range(steps + 1):
        z = start_z + (end_z - start_z) * (i / steps)
        bot.move_position(0, 0, int(z))
        time.sleep(0.05)

    print("Hexapod is now standing.")

if __name__ == "__main__":
    stand_up()
