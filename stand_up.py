# stand_up.py
from control import Control
import time

def stand_up():
    bot = Control()

    print("Starting corrected stand-up sequence...")
    time.sleep(1)


    start_z = -80    # crouched
    end_z   = -30    # normal standing height (higher body)

    steps = 25
    for step in range(steps):
        z = start_z + (end_z - start_z) * (step / steps)
        bot.move_position(0, 0, int(z))
        time.sleep(0.05)

    print("Hexapod should now be standing at normal height.")

if __name__ == "__main__":
    stand_up()
