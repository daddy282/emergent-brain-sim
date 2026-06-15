from .base_region import BrainRegion, Signal


class VTA(BrainRegion):
    """
    The Ventral Tegmental Area (VTA) — dopamine production, reward
    prediction error, motivation signaling.

    Biological rules implemented:
    - Reward prediction error: fires more for unexpected reward, less for
      expected/predicted reward (the core dopamine learning signal)
    - Dopamine output: produces the global dopamine level other regions read
    - Novelty-driven firing: hippocampal novelty signals boost dopamine release
    - Tolerance: repeated identical rewards reduce response over time
      (mirrors dopaminergic downregulation / tolerance)
    - Stress suppression: high cortisol blunts dopamine release
    """

    def __init__(self, instance_id: int = 0):
        super().__init__(
            region_id=f"vta_{instance_id}",
            name="VTA",
            instance_id=instance_id
        )

        # Dopamine output (this region produces the global dopamine level)
        self.dopamine_output: float = 0.5  # baseline dopamine

        # Reward prediction
        self.expected_reward: float = 0.3   # running estimate of expected reward
        self.prediction_error: float = 0.0  # actual - expected

        # Tolerance — repeated rewards reduce response
        self.reward_history: dict = {}      # reward_id -> exposure_count

        # Novelty boost from hippocampus
        self.pending_novelty_boost: float = 0.0

        self.threshold = 0.3
        self.baseline_activation = 0.15

        self.sensitivity["cortisol"] = 0.7   # stress blunts dopamine
        self.sensitivity["glutamate"] = 0.8

        # Anatomical position (midbrain, ventral)
        self.position = {"x": 0.0, "y": -0.3, "z": -0.1}

        # Learning rate for reward expectation
        self.EXPECTATION_LEARNING_RATE = 0.05

    def process(self, neuromodulators: dict, dt: float):
        cortisol = neuromodulators.get("cortisol", 0.3)

        # 1. Apply novelty boost from hippocampus
        if self.pending_novelty_boost > 0:
            self.activation += self.pending_novelty_boost * 0.3 * dt
            self.pending_novelty_boost = max(0.0, self.pending_novelty_boost - 0.5 * dt)

        # 2. Chronic stress blunts dopamine release
        if cortisol > 0.6:
            self.activation -= (cortisol - 0.6) * 0.2 * dt

        # 3. Compute dopamine output from prediction error + activation
        # Positive prediction error -> dopamine burst, negative -> dip below baseline
        base_output = 0.5 + self.prediction_error * 0.5
        self.dopamine_output += (base_output - self.dopamine_output) * 0.3 * dt
        self.dopamine_output = max(0.0, min(1.0, self.dopamine_output))

        # 4. Prediction error decays back toward zero (event has been "processed")
        self.prediction_error *= (1.0 - 0.3 * dt)

        # 5. Expected reward slowly relaxes toward neutral if nothing happens
        self.expected_reward += (0.3 - self.expected_reward) * 0.05 * dt

    def receive_reward(self, reward_id: str, reward_value: float):
        """
        Called when a rewarding event occurs.
        Computes reward prediction error and applies tolerance.
        """
        exposures = self.reward_history.get(reward_id, 0)
        tolerance_factor = max(0.2, 1.0 - (exposures * 0.1))

        effective_reward = reward_value * tolerance_factor

        # Prediction error = actual - expected
        self.prediction_error = effective_reward - self.expected_reward

        # Activation spikes proportional to positive prediction error
        if self.prediction_error > 0:
            self.activation = min(1.0, self.activation + self.prediction_error * 0.6)

        # Update expectation toward this reward (learning)
        self.expected_reward += self.EXPECTATION_LEARNING_RATE * self.prediction_error

        # Track tolerance
        self.reward_history[reward_id] = exposures + 1

    def receive_novelty_signal(self, novelty: float):
        """Called when hippocampus reports high novelty — boosts dopamine firing."""
        self.pending_novelty_boost = max(self.pending_novelty_boost, novelty)

    def get_neuromodulator_output(self) -> dict:
        """Called by the simulation loop to update the global dopamine pool."""
        return {"dopamine": self.dopamine_output}

    def get_output_signals(self) -> list[Signal]:
        strength = self.activation

        return [
            Signal(
                source_id=self.region_id,
                signal_type="dopamine",
                strength=strength * self.dopamine_output,
                metadata={"target_hint": "prefrontal", "purpose": "motivation_signal",
                          "prediction_error": round(self.prediction_error, 4)}
            ),
            Signal(
                source_id=self.region_id,
                signal_type="dopamine",
                strength=strength * 0.7,
                metadata={"target_hint": "accumbens", "purpose": "reward_signal",
                          "prediction_error": round(self.prediction_error, 4)}
            ),
        ]

    def _get_refractory_period(self) -> float:
        return 0.05

    def get_state(self) -> dict:
        state = super().get_state()
        state.update({
            "dopamine_output": round(self.dopamine_output, 4),
            "expected_reward": round(self.expected_reward, 4),
            "prediction_error": round(self.prediction_error, 4),
            "rewards_tracked": len(self.reward_history),
        })
        return state