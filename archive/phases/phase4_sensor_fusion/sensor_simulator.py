import random
import time

class SensorSimulator:
    def __init__(self):
        self.baseline_temp = 24.0      # normal room temp (C)
        self.baseline_smoke = 5.0      # normal smoke particle level (arbitrary unit)
        self.baseline_gas = 10.0       # normal gas level (arbitrary unit)
        self.event_active = False
        self.event_ticks_left = 0

    def maybe_trigger_event(self):
        # small random chance each reading that a "fire event" starts
        if not self.event_active and random.random() < 0.03:
            self.event_active = True
            self.event_ticks_left = random.randint(8, 15)
            print(">>> SIMULATED EVENT STARTED (fire-like sensor spike) <<<")

    def read(self):
        self.maybe_trigger_event()

        if self.event_active:
            # during an event, values climb sharply
            temp = self.baseline_temp + random.uniform(20, 40)
            smoke = self.baseline_smoke + random.uniform(40, 80)
            gas = self.baseline_gas + random.uniform(30, 60)

            self.event_ticks_left -= 1
            if self.event_ticks_left <= 0:
                self.event_active = False
                print(">>> SIMULATED EVENT ENDED <<<")
        else:
            # normal fluctuation
            temp = self.baseline_temp + random.uniform(-1, 1)
            smoke = self.baseline_smoke + random.uniform(-1, 1)
            gas = self.baseline_gas + random.uniform(-2, 2)

        return {
            "temperature": round(temp, 1),
            "smoke_level": round(smoke, 1),
            "gas_level": round(gas, 1),
        }


if __name__ == "__main__":
    sim = SensorSimulator()
    print("Starting sensor simulation. Press Ctrl+C to stop.\n")
    try:
        while True:
            reading = sim.read()
            print(f"Temp: {reading['temperature']} C | Smoke: {reading['smoke_level']} | Gas: {reading['gas_level']}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nSimulation stopped.")
