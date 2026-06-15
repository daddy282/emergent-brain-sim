from .base_region import BrainRegion, Signal


class Amygdala(BrainRegion):
    """
    The amygdala — threat detection and emotional memory tagging.
    
    Biological rules implemented:
    - Fast dirty processing: reacts to threat signals before full cortical analysis
    - Emotional tagging: amplifies signals to hippocampus when threat is high
    - Fear conditioning: lowers threshold when paired stimuli repeat
    - Inhibited by prefrontal cortex under low threat (top-down regulation)
    - Drives hypothalamus for fight-or-flight output
    - High dopamine dysregulation → hypervigilance (salience misattribution)
    """

    def __init__(self, instance_id: int = 0, side: str = "bilateral"):
        super().__init__(
            region_id=f"amygdala_{instance_id}",
            name="Amygdala",
            instance_id=instance_id
        )
        self.side = side

        # Amygdala-specific state
        self.threat_level: float = 0.0          # current perceived threat 0-1
        self.fear_memory_strength: float = 0.0  # conditioned fear accumulation
        self.pfc_inhibition: float = 0.0        # suppression from prefrontal cortex
        self.emotional_valence: float = 0.0     # -1 (negative) to +1 (positive)

        # Lower threshold than most regions — reacts fast
        self.threshold = 0.2
        self.baseline_activation = 0.15

        # High sensitivity to norepinephrine (threat arousal signal)
        self.sensitivity["norepinephrine"] = 0.9
        self.sensitivity["dopamine"] = 0.8      # dopamine dysregulation → hypervigilance
        self.sensitivity["cortisol"] = 0.7      # stress amplifies amygdala

        # Anatomical position (left amygdala, approximate MNI coordinates normalized)
        self.position = {"x": -0.22, "y": -0.05, "z": -0.18}

        # Fear conditioning history: list of (conditioned_stimulus_id, strength)
        self.conditioned_fears: dict = {}

    def process(self, neuromodulators: dict, dt: float):
        """
        Amygdala-specific processing:
        1. Apply PFC top-down inhibition
        2. Boost activation if dopamine is dysregulated (misattributed salience)
        3. Update threat level based on activation
        4. Strengthen fear conditioning if threshold exceeded repeatedly
        """
        dopamine = neuromodulators.get("dopamine", 0.5)
        cortisol = neuromodulators.get("cortisol", 0.3)

        # 1. PFC top-down inhibition — prefrontal cortex can suppress amygdala
        if self.pfc_inhibition > 0:
            self.activation -= self.pfc_inhibition * 0.4 * dt
            self.pfc_inhibition = max(0.0, self.pfc_inhibition - 0.1 * dt)

        # 2. Dopamine dysregulation → hypervigilance
        # When dopamine is too high OR too low, amygdala misattributes salience
        dopamine_dysregulation = abs(dopamine - 0.5) * 2.0
        if dopamine_dysregulation > 0.3:
            self.activation += dopamine_dysregulation * 0.15 * dt

        # 3. Cortisol amplification — chronic stress keeps amygdala hot
        self.activation += (cortisol - 0.3) * 0.1 * dt

        # 4. Update threat level (smoothed)
        self.threat_level += (self.activation - self.threat_level) * 0.3 * dt

        # 5. Fear conditioning: if repeatedly activated above threshold,
        # lower the threshold slightly (sensitization)
        if self.activation > self.threshold and len(self.activation_history) > 10:
            recent_avg = sum(list(self.activation_history)[-10:]) / 10
            if recent_avg > 0.5:
                self.threshold = max(0.1, self.threshold - 0.001 * dt)
                self.fear_memory_strength = min(1.0, self.fear_memory_strength + 0.002 * dt)

    def receive_pfc_inhibition(self, strength: float):
        """Called by prefrontal cortex to apply top-down regulation."""
        self.pfc_inhibition = min(1.0, self.pfc_inhibition + strength)

    def get_output_signals(self) -> list[Signal]:
        """
        When amygdala fires it sends:
        - Strong glutamate to hypothalamus (fight-or-flight cascade)
        - Glutamate to hippocampus (emotional memory tagging — tag this memory as important)
        - Glutamate to ACC (conflict/alarm signal)
        - Glutamate to insula (body-state awareness)
        - Weak norepinephrine-like arousal signal to thalamus
        """
        strength = self.activation

        return [
            Signal(
                source_id=self.region_id,
                signal_type="glutamate",
                strength=strength * 0.9,
                metadata={"target_hint": "hypothalamus", "purpose": "fight_flight"}
            ),
            Signal(
                source_id=self.region_id,
                signal_type="glutamate",
                strength=strength * 0.7,
                metadata={"target_hint": "hippocampus", "purpose": "emotional_tag",
                          "valence": self.emotional_valence, "threat": self.threat_level}
            ),
            Signal(
                source_id=self.region_id,
                signal_type="glutamate",
                strength=strength * 0.6,
                metadata={"target_hint": "acc", "purpose": "alarm"}
            ),
            Signal(
                source_id=self.region_id,
                signal_type="glutamate",
                strength=strength * 0.5,
                metadata={"target_hint": "insula", "purpose": "body_awareness"}
            ),
        ]

    def _get_refractory_period(self) -> float:
        """Amygdala recovers quickly — it needs to keep monitoring for threats."""
        return 0.02

    def get_state(self) -> dict:
        state = super().get_state()
        state.update({
            "threat_level": round(self.threat_level, 4),
            "fear_memory_strength": round(self.fear_memory_strength, 4),
            "pfc_inhibition": round(self.pfc_inhibition, 4),
            "emotional_valence": round(self.emotional_valence, 4),
            "conditioned_fears": len(self.conditioned_fears),
        })
        return state