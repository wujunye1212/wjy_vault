import tkinter as tk
from time import strftime
from pytz import timezone
import pytz


class DigitalClock:
    def __init__(self, root):
        self.root = root
        self.root.title("Digital Clock with Time Zones")

        # Timezone list
        self.timezones = [
            ("UTC", timezone("UTC")),
            ("New York", timezone("America/New_York")),
            ("London", timezone("Europe/London")),
            ("Tokyo", timezone("Asia/Tokyo")),
            ("Sydney", timezone("Australia/Sydney")),
        ]

        # Create labels for each timezone
        self.time_labels = []
        for tz_name, _ in self.timezones:
            label = tk.Label(root, font=("Helvetica", 18), background="black", foreground="white")
            label.pack(anchor="center", pady=5)
            self.time_labels.append((tz_name, label))

        self.update_clock()

    def update_clock(self):
        for (tz_name, tz), (_, label) in zip(self.timezones, self.time_labels):
            current_time = strftime("%Y-%m-%d %H:%M:%S", tz.localize(pytz.utc).utcoffset())
            label.config(text=f"{tz_name}: {current_time}")
        
        # Schedule the update method to be called every second
        self.root.after(1000, self.update_clock)


if __name__ == "__main__":
    root = tk.Tk()
    clock = DigitalClock(root)
    root.mainloop()