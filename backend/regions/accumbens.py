from .base_region import BrainRegion, Signal


class NucleusAccumbens(BrainRegion):
    """
    Nucleus Accumbens — reward consumption, wanting vs liking, motivation
    to act, addiction-relevant reinforcement loop.

    Biological rules implemented:
    - Wanting vs liking: dopamine drives "wanting" (motivation/craving),
      separate from the "liking" (hedonic) response which fades faster
    - Reinforcement loop: repeated reward signals strengthen a craving
      baseline (sensitization) — models addiction-like escalation
    - Action drive: high activation produces an "approach" output signal
      that other regions could use to bias behavior toward reward-seeking
    - Satiation: liking response diminishes with repeated consumption in
      a short window, independent of dopamine wanting signal
    - PFC regulation: like the amygdala, can be dampened by prefrontal
      top-down control (impulse regulation)
    """

    def __init__(self, instance_id: int = 0):
        super().__init__(
            region_id=f"accumbens_{instance_id}",
            name="Nucleus Accumbens",
            instance_id=instance_id
        )

        # Wanting vs liking
        self.wanting: float = 0.0       # craving/motivation, driven by dopamine
        self.liking: float = 0.0        # hedonic response, fades with satiation

        # Sensitization (addiction-like escalation of wanting baseline)
        self.sensitization: float = 0.0
        self.consumption_count: int = 0
        self.satiation: float = 0.0

        # Action drive output
        self.approach_drive: float = 0.0

        # PFC regulation
        self.pfc_inhibition: float = 0.0

        self.threshold = 0.3
        self.baseline_activation = 0.1

        self.sensitivity["dopamine"] = 1.0  # primary driver
        self.sensitivity["gaba"] = 1.0

        # Anatomical position (ventral, anterior, near VTA)
        self.position = {"x": 0.0, "y": -0.2, "z": 0.0}

    def process(self, neuromodulators: dict, dt: float):
        dopamine = neuromodulators.get("dopamine", 0.5)

        # 1. PFC inhibition dampens activation (impulse regulation)
        if self.pfc_inhibition > 0:
            self.activation -= self.pfc_inhibition * 0.3 * dt
            self.pfc_inhibition = max(0.0, self.pfc_inhibition - 0.1 * dt)

        # 2. Wanting tracks dopamine level, boosted by sensitization
        dopamine_drive = max(0.0, dopamine - 0.5) * 2.0  # only above-baseline dopamine drives wanting
        target_wanting = dopamine_drive * (1.0 + self.sensitization)
        self.wanting += (min(1.0, target_wanting) - self.wanting) * 0.3 * dt

        # 3. Liking fades with satiation, independent of dopamine
        self.liking *= (1.0 - 0.4 * dt)
        self.satiation = max(0.0, self.satiation - 0.05 * dt)

        # 4. Wanting drives activation more than liking does
        # (the "wanting > liking" divergence seen in addiction)
        self.activation += (self.wanting * 0.4 + self.liking * 0.2) * dt

        # 5. Approach drive tracks wanting, with momentum
        self.approach_drive += (self.wanting - self.approach_drive) * 0.25 * dt

    def receive_signal(self, signal: Signal):
        """Override to capture reward signals and apply consumption/satiation."""
        super().receive_signal(signal)

        if signal.metadata.get("purpose") == "reward_signal":
            self._consume_reward(signal.strength)

    def _consume_reward(self, strength: float):
        """
        A reward was consumed. Liking response is reduced by current
        satiation level. Repeated consumption builds sensitization
        (wanting baseline creeps up even as liking diminishes).
        """
        satiation_discount = max(0.2, 1.0 - self.satiation)
        self.liking = min(1.0, self.liking + strength * satiation_discount)

        self.satiation = min(1.0, self.satiation + 0.15)
        self.consumption_count += 1

        # Sensitization grows slowly with repeated consumption
        self.sensitization = min(1.0, self.sensitization + 0.01)

    def receive_pfc_inhibition(self, strength: float):
        """Called by prefrontal cortex for impulse regulation."""
        self.pfc_inhibition = min(1.0, self.pfc_inhibition + strength)

    def get_output_signals(self) -> list[Signal]:
        strength = self.activation

        return [
            Signal(
                source_id=self.region_id,
                signal_type="glutamate",
                strength=strength * self.approach_drive,
                metadata={"target_hint": "prefrontal", "purpose": "approach_drive",
                          "wanting": round(self.wanting, 4),
                          "liking": round(self.liking, 4)}
            ),
        ]

    def _get_refractory_period(self) -> float:
        return 0.05

    def get_state(self) -> dict:
        state = super().get_state()
        state.update({
            "wanting": round(self.wanting, 4),
            "liking": round(self.liking, 4),
            "sensitization": round(self.sensitization, 4),
            "satiation": round(self.satiation, 4),
            "approach_drive": round(self.approach_drive, 4),
            "consumption_count": self.consumption_count,
        })
        return state