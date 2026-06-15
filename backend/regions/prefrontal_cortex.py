import time
from collections import deque
from .base_region import BrainRegion, Signal


class PrefrontalCortex(BrainRegion):
    """
    The prefrontal cortex — executive control, regulation, planning, inhibition.

    Biological rules implemented:
    - Top-down inhibition: dampens amygdala overactivation (emotional regulation)
    - Working memory: holds recent context from hippocampus for decision-making
    - Cognitive load: too many concurrent inputs degrades regulation capacity
    - Slow maturation: regulatory strength ramps up gradually over ticks
      (mirrors PFC being the last region to fully develop)
    - Dopamine-dependent function: PFC efficiency follows an inverted-U with dopamine
    - Stress vulnerability: high cortisol impairs executive function ("PFC goes offline")
    """

    def __init__(self, instance_id: int = 0):
        super().__init__(
            region_id=f"prefrontal_{instance_id}",
            name="Prefrontal Cortex",
            instance_id=instance_id
        )

        # Working memory — holds recent contextual signals
        self.working_memory: deque = deque(maxlen=20)

        # Regulation state
        self.regulation_strength: float = 0.0     # 0-1, ability to inhibit amygdala
        self.maturation: float = 0.0               # ramps up over time, caps regulation
        self.cognitive_load: float = 0.0           # rises with input volume

        # Incoming context from hippocampus
        self.episodic_context: dict = {}
        self.context_novelty: float = 0.0

        # Stress-induced impairment
        self.offline: bool = False                 # PFC "goes offline" under extreme stress
        self.offline_timer: float = 0.0

        # Anatomical position (anterior, superior)
        self.position = {"x": 0.0, "y": 0.35, "z": 0.22}

        self.threshold = 0.35
        self.baseline_activation = 0.15
        self.sensitivity["cortisol"] = 0.8         # highly stress-sensitive
        self.sensitivity["dopamine"] = 0.7
        self.sensitivity["norepinephrine"] = 0.6

        # Maturation rate — PFC regulatory capacity builds slowly
        self.MATURATION_RATE = 0.0008

    def process(self, neuromodulators: dict, dt: float):
        cortisol = neuromodulators.get("cortisol", 0.3)
        dopamine = neuromodulators.get("dopamine", 0.5)

        # 1. Maturation ramps up over time (caps at 1.0)
        self.maturation = min(1.0, self.maturation + self.MATURATION_RATE * dt)

        # 2. Acute stress can push PFC offline
        if cortisol > 0.85:
            self.offline = True
            self.offline_timer = 2.0  # seconds of impaired function
        if self.offline_timer > 0:
            self.offline_timer = max(0.0, self.offline_timer - dt)
            if self.offline_timer == 0:
                self.offline = False

        if self.offline:
            # Executive function suppressed — activation collapses
            self.activation *= (1.0 - 0.3 * dt)
            self.regulation_strength = 0.0
            return

        # 3. Dopamine-dependent efficiency — inverted-U (optimal ~0.6)
        dopamine_efficiency = 1.0 - abs(dopamine - 0.6) * 1.2
        dopamine_efficiency = max(0.15, dopamine_efficiency)

        # 4. Cognitive load reduces regulation capacity
        load_penalty = max(0.0, 1.0 - self.cognitive_load * 0.6)

        # 5. Compute regulation strength from maturation, dopamine, load, chronic stress
        chronic_stress_penalty = max(0.0, 1.0 - cortisol * 0.5)
        self.regulation_strength = (
            self.maturation * dopamine_efficiency * load_penalty * chronic_stress_penalty
        )
        self.regulation_strength = max(0.0, min(1.0, self.regulation_strength))

        # 6. Working memory contextual boost — episodic context sustains activation
        if self.episodic_context:
            context_boost = self.episodic_context.get("emotional_weight", 0.0) * 0.2
            self.activation += context_boost * dt

        # 7. Cognitive load decays naturally
        self.cognitive_load *= (1.0 - 0.1 * dt)

    def receive_signal(self, signal: Signal):
        """Override to track cognitive load and capture episodic context."""
        super().receive_signal(signal)
        self.cognitive_load = min(1.0, self.cognitive_load + 0.05)

        if signal.metadata.get("purpose") == "episodic_context":
            self.episodic_context = {
                "memory_count": signal.metadata.get("memory_count", 0),
                "novelty": signal.metadata.get("novelty", 0.0),
                "emotional_weight": signal.strength,
                "source": signal.source_id,
                "timestamp": signal.timestamp,
            }
            self.context_novelty = signal.metadata.get("novelty", 0.0)
            self.working_memory.append(self.episodic_context.copy())

    def get_output_signals(self) -> list[Signal]:
        """
        Prefrontal outputs:
        - To amygdala: GABA inhibitory signal (top-down regulation)
        - To hippocampus: glutamate signal reflecting executive attention/encoding priority
        """
        signals = []

        # Regulatory inhibition of amygdala — scales with regulation_strength
        if self.regulation_strength > 0.05:
            signals.append(Signal(
                source_id=self.region_id,
                signal_type="gaba",
                strength=self.activation * self.regulation_strength,
                metadata={"target_hint": "amygdala", "purpose": "emotional_regulation",
                          "regulation_strength": round(self.regulation_strength, 4)}
            ))

        # Executive attention signal to hippocampus
        signals.append(Signal(
            source_id=self.region_id,
            signal_type="glutamate",
            strength=self.activation * 0.5,
            metadata={"target_hint": "hippocampus", "purpose": "attention_priority",
                      "maturation": round(self.maturation, 4)}
        ))

        return signals

    def _get_refractory_period(self) -> float:
        return 0.08  # slightly longer than default — PFC processing is comparatively slow

    def get_state(self) -> dict:
        state = super().get_state()
        state.update({
            "regulation_strength": round(self.regulation_strength, 4),
            "maturation": round(self.maturation, 4),
            "cognitive_load": round(self.cognitive_load, 4),
            "offline": self.offline,
            "working_memory_size": len(self.working_memory),
            "context_novelty": round(self.context_novelty, 4),
        })
        return state