import random
import time

class SensorSimulator:
    def __init__(self):
        self.baseline = {"temperature": 24.0, "smoke_level": 5.0, "gas_level": 10.0}
        self.current = dict(self.baseline)
        self.target = dict(self.baseline)

        self.event_active = False
        self.cooldown_until = 0       # timestamp - no new random event before this
        self.ramp_speed = 0.15        # how fast values move toward target each update (0-1)

    def trigger_event_manually(self):
        """Force a fire-like event to start right now (for controlled testing)."""
        self.event_active = True
        self.target = {"temperature": 58.0, "smoke_level": 65.0, "gas_level": 42.0}
        print(">>> MANUAL EVENT TRIGGERED <<<")

    def end_event(self):
        self.event_active = False
        self.target = dict(self.baseline)
        self.cooldown_until = time.time() + 20   # 20s cooldown before another random event
        print(">>> EVENT ENDED, returning to baseline <<<")

    def maybe_trigger_random_event(self):
        if self.event_active:
            return
        if time.time() < self.cooldown_until:
            return
        # much lower chance, checked once per second in practice
        if random.random() < 0.01:
            self.trigger_event_manually()

    def read(self):
        self.maybe_trigger_random_event()

        # smoothly move current values toward target (gradual ramp, not instant jump)
        for key in self.current:
            diff = self.target[key] - self.current[key]
            self.current[key] += diff * self.ramp_speed

            # small realistic noise on top
            noise = random.uniform(-0.15, 0.15)
            self.current[key] += noise

        # if we were in an event and we're close enough to target, hold briefly then end
        if self.event_active:
            close_enough = all(
                abs(self.current[k] - self.target[k]) < 2.0 for k in self.current
            )
            if close_enough:
                # hold at peak for a few reads before ramping back down
                if not hasattr(self, "_hold_counter"):
                    self._hold_counter = 0
                self._hold_counter += 1
                if self._hold_counter > 5:
                    self._hold_counter = 0
                    self.end_event()

        return {
            "temperature": round(self.current["temperature"], 1),
            "smoke_level": round(max(self.current["smoke_level"], 0), 1),
            "gas_level": round(max(self.current["gas_level"], 0), 1),
        }


if __name__ == "__main__":
    sim = SensorSimulator()
    print("Starting sensor simulation. Press Ctrl+C to stop.")
    print("(Random events are rare and gradual now - values ramp smoothly, not instant jumps)\n")
    try:
        while True:
            reading = sim.read()
            print(f"Temp: {reading['temperature']} C | Smoke: {reading['smoke_level']} | Gas: {reading['gas_level']}")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nSimulation stopped.")
