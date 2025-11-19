# stand_up.py
from control import Control
import time

def stand_up():
    bot = Control()

    print("Starting stand-up sequence...")
    time.sleep(1)

    # Raise body height gradually from crouched (-30) to higher (-90)
    target_height = -90   # adjust if needed
    steps = 20

    for step in range(steps):
        z = int((target_height + 30) * (step / steps))  # from 0 to target offset
        bot.move_position(0, 0, z)  # x=0, y=0, z controls body height
        time.sleep(0.05)

    print("Hexapod is now standing up.")

if __name__ == "__main__":
    stand_up()
