from .base_region import BrainRegion, Signal


class Hypothalamus(BrainRegion):
    """
    The hypothalamus — homeostatic control center, stress axis (HPA),
    fight-or-flight execution.

    Biological rules implemented:
    - HPA axis: receives amygdala threat signals, releases cortisol
    - Fight-or-flight: high activation triggers norepinephrine surge
    - Homeostasis: slowly pulls cortisol/arousal back toward baseline
    - Chronic activation: prolonged high activation raises baseline cortisol
      (allostatic load — chronic stress recalibrates the resting state)
    - Negative feedback: high cortisol eventually self-suppresses hypothalamus
      (mimics cortisol's negative feedback on the HPA axis)
    """

    def __init__(self, instance_id: int = 0):
        super().__init__(
            region_id=f"hypothalamus_{instance_id}",
            name="Hypothalamus",
            instance_id=instance_id
        )

        # Neuromodulator output levels (this region produces these globally)
        self.cortisol_output: float = 0.3       # baseline cortisol release
        self.norepinephrine_output: float = 0.4 # baseline arousal release

        # Allostatic load — chronic stress shifts baseline upward permanently
        self.allostatic_load: float = 0.0

        # Fight-or-flight state
        self.fight_flight_active: bool = False
        self.fight_flight_timer: float = 0.0

        self.threshold = 0.3
        self.baseline_activation = 0.15

        # Very sensitive to glutamate from amygdala (threat cascade)
        self.sensitivity["glutamate"] = 1.0
        self.sensitivity["cortisol"] = 0.6  # negative feedback sensitivity

        # Anatomical position (central, ventral)
        self.position = {"x": 0.0, "y": -0.15, "z": 0.05}

    def process(self, neuromodulators: dict, dt: float):
        cortisol = neuromodulators.get("cortisol", 0.3)

        # 1. Fight-or-flight trigger
        if self.activation > 0.5 and not self.fight_flight_active:
            self.fight_flight_active = True
            self.fight_flight_timer = 1.5

        if self.fight_flight_active:
            self.fight_flight_timer = max(0.0, self.fight_flight_timer - dt)
            self.norepinephrine_output = min(1.0, self.norepinephrine_output + 0.3 * dt)
            self.cortisol_output = min(1.0, self.cortisol_output + 0.2 * dt)
            if self.fight_flight_timer == 0:
                self.fight_flight_active = False

        # 2. Negative feedback — high circulating cortisol suppresses hypothalamus
        if cortisol > 0.7:
            self.activation -= (cortisol - 0.7) * 0.3 * dt

        # 3. Chronic activation raises allostatic load (long-term baseline shift)
        if self.activation > 0.6:
            self.allostatic_load = min(1.0, self.allostatic_load + 0.0003 * dt)

        # Allostatic load raises resting cortisol/baseline permanently
        self.baseline_activation = 0.15 + self.allostatic_load * 0.2
        self.cortisol_output = max(0.2, self.cortisol_output - 0.05 * dt) + self.allostatic_load * 0.1
        self.cortisol_output = min(1.0, self.cortisol_output)

        # 4. Homeostatic pull — norepinephrine output relaxes toward baseline
        if not self.fight_flight_active:
            self.norepinephrine_output += (0.4 - self.norepinephrine_output) * 0.2 * dt

        # Clamp outputs
        self.cortisol_output = max(0.0, min(1.0, self.cortisol_output))
        self.norepinephrine_output = max(0.0, min(1.0, self.norepinephrine_output))

    def get_neuromodulator_output(self) -> dict:
        """
        Called by the simulation loop to update global neuromodulator pools.
        This is how the hypothalamus influences every other region's
        neuromodulator-dependent processing.
        """
        return {
            "cortisol": self.cortisol_output,
            "norepinephrine": self.norepinephrine_output,
        }

    def get_output_signals(self) -> list[Signal]:
        strength = self.activation

        return [
            Signal(
                source_id=self.region_id,
                signal_type="norepinephrine",
                strength=strength * self.norepinephrine_output,
                metadata={"target_hint": "amygdala", "purpose": "arousal_feedback"}
            ),
            Signal(
                source_id=self.region_id,
                signal_type="glutamate",
                strength=strength * 0.6,
                metadata={"target_hint": "insula", "purpose": "body_state",
                          "fight_flight": self.fight_flight_active}
            ),
        ]

    def _get_refractory_period(self) -> float:
        return 0.1

    def get_state(self) -> dict:
        state = super().get_state()
        state.update({
            "cortisol_output": round(self.cortisol_output, 4),
            "norepinephrine_output": round(self.norepinephrine_output, 4),
            "allostatic_load": round(self.allostatic_load, 4),
            "fight_flight_active": self.fight_flight_active,
        })
        return state