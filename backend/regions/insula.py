from .base_region import BrainRegion, Signal


class Insula(BrainRegion):
    """
    Insula — interoception (body-state awareness), disgust, emotional
    salience integration, gut-feeling signals.

    Biological rules implemented:
    - Interoceptive integration: combines amygdala body-awareness signals
      and hypothalamus fight-or-flight signals into a single "felt state"
    - Salience mapping: tracks how strongly the body-state should grab
      attention (interrupts ongoing processing if high)
    - Disgust/aversion: a distinct negative-valence channel separate from
      fear, triggered by strong negative body-state signals
    - Habituation to chronic states: sustained high arousal reduces the
      felt intensity over time (interoceptive habituation, common in
      chronic anxiety/dissociation)
    - Gut-feeling output: sends a slow-building "intuition" signal to
      prefrontal cortex that biases decisions without explicit reasoning
    """

    def __init__(self, instance_id: int = 0):
        super().__init__(
            region_id=f"insula_{instance_id}",
            name="Insula",
            instance_id=instance_id
        )

        # Interoceptive state
        self.body_state_intensity: float = 0.0   # how strong the felt body signal is
        self.felt_state_habituation: float = 0.0  # 0-1, reduces felt intensity over time

        # Disgust/aversion channel
        self.aversion_level: float = 0.0

        # Salience — how much this should interrupt other processing
        self.salience: float = 0.0

        # Gut feeling — slow-accumulating bias signal
        self.gut_feeling: float = 0.0

        self.threshold = 0.3
        self.baseline_activation = 0.1

        self.sensitivity["glutamate"] = 0.9
        self.sensitivity["cortisol"] = 0.5

        # Anatomical position (lateral, deep)
        self.position = {"x": 0.3, "y": -0.05, "z": -0.05}

    def process(self, neuromodulators: dict, dt: float):
        cortisol = neuromodulators.get("cortisol", 0.3)

        # 1. Body-state intensity feeds activation, dampened by habituation
        effective_intensity = self.body_state_intensity * (1.0 - self.felt_state_habituation)
        self.activation += effective_intensity * 0.3 * dt

        # 2. Chronic high activation builds habituation (interoceptive blunting)
        if self.activation > 0.6:
            self.felt_state_habituation = min(0.8, self.felt_state_habituation + 0.0004 * dt)
        else:
            self.felt_state_habituation = max(0.0, self.felt_state_habituation - 0.0002 * dt)

        # 3. Salience tracks activation but spikes faster, decays faster
        target_salience = self.activation
        self.salience += (target_salience - self.salience) * 0.4 * dt

        # 4. Cortisol amplifies aversion (stress makes things feel worse)
        if self.aversion_level > 0:
            self.activation += self.aversion_level * (0.5 + cortisol * 0.5) * 0.2 * dt
            self.aversion_level *= (1.0 - 0.3 * dt)

        # 5. Gut feeling slowly accumulates from sustained activation,
        # slowly decays otherwise — represents accumulated "felt sense"
        if self.activation > 0.4:
            self.gut_feeling = min(1.0, self.gut_feeling + 0.02 * dt)
        else:
            self.gut_feeling = max(0.0, self.gut_feeling - 0.01 * dt)

        # 6. Body state intensity itself decays each tick (re-driven by signals)
        self.body_state_intensity *= (1.0 - 0.4 * dt)

    def receive_signal(self, signal: Signal):
        """
        Override to extract body-state intensity and detect aversion triggers
        from amygdala / hypothalamus signals.
        """
        super().receive_signal(signal)

        purpose = signal.metadata.get("purpose")

        if purpose == "body_awareness":
            self.body_state_intensity = min(1.0, self.body_state_intensity + signal.strength)

        if purpose == "body_state":
            self.body_state_intensity = min(1.0, self.body_state_intensity + signal.strength)
            if signal.metadata.get("fight_flight"):
                self.aversion_level = min(1.0, self.aversion_level + signal.strength * 0.5)

    def get_output_signals(self) -> list[Signal]:
        strength = self.activation

        return [
            Signal(
                source_id=self.region_id,
                signal_type="glutamate",
                strength=strength * self.salience,
                metadata={"target_hint": "acc", "purpose": "salience_alert",
                          "salience": round(self.salience, 4)}
            ),
            Signal(
                source_id=self.region_id,
                signal_type="glutamate",
                strength=self.gut_feeling * 0.6,
                metadata={"target_hint": "prefrontal", "purpose": "gut_feeling",
                          "aversion": round(self.aversion_level, 4)}
            ),
        ]

    def _get_refractory_period(self) -> float:
        return 0.06

    def get_state(self) -> dict:
        state = super().get_state()
        state.update({
            "body_state_intensity": round(self.body_state_intensity, 4),
            "felt_state_habituation": round(self.felt_state_habituation, 4),
            "aversion_level": round(self.aversion_level, 4),
            "salience": round(self.salience, 4),
            "gut_feeling": round(self.gut_feeling, 4),
        })
        return state