from .base_region import BrainRegion, Signal


class ACC(BrainRegion):
    """
    Anterior Cingulate Cortex — conflict monitoring, error detection,
    alarm signaling, effort allocation.

    Biological rules implemented:
    - Conflict detection: fires when competing signals arrive close together
      (e.g. amygdala alarm + PFC regulation at the same time = conflict)
    - Error monitoring: tracks mismatches between expected and actual outcomes
    - Alarm relay: amplifies amygdala alarm signals toward prefrontal cortex
      to recruit executive control
    - Effort signaling: sustained conflict raises an "effort" output that
      modulates prefrontal engagement
    - Habituation: repeated identical conflicts reduce ACC response over time
    """

    def __init__(self, instance_id: int = 0):
        super().__init__(
            region_id=f"acc_{instance_id}",
            name="ACC",
            instance_id=instance_id
        )

        # Conflict tracking
        self.conflict_level: float = 0.0
        self.recent_signal_sources: dict = {}   # source_id -> last_seen_timestamp
        self.effort_output: float = 0.0

        # Error monitoring
        self.error_signal: float = 0.0

        # Habituation to repeated conflicts
        self.conflict_habituation: dict = {}    # conflict_signature -> exposure_count

        self.threshold = 0.25
        self.baseline_activation = 0.12

        self.sensitivity["glutamate"] = 0.9
        self.sensitivity["norepinephrine"] = 0.6

        # Anatomical position (medial, superior, anterior)
        self.position = {"x": 0.0, "y": 0.2, "z": 0.1}

        # How close in time two signals must arrive to count as "conflict"
        self.CONFLICT_WINDOW = 0.3

    def process(self, neuromodulators: dict, dt: float):
        norepinephrine = neuromodulators.get("norepinephrine", 0.4)

        # 1. Conflict amplifies activation, scaled by arousal
        if self.conflict_level > 0:
            self.activation += self.conflict_level * 0.3 * (0.5 + norepinephrine) * dt

        # 2. Error signal contributes to activation, then decays
        if self.error_signal > 0:
            self.activation += self.error_signal * 0.25 * dt
            self.error_signal *= (1.0 - 0.4 * dt)

        # 3. Effort output builds with sustained conflict, decays without it
        if self.conflict_level > 0.3:
            self.effort_output = min(1.0, self.effort_output + 0.1 * dt)
        else:
            self.effort_output = max(0.0, self.effort_output - 0.05 * dt)

        # 4. Conflict level decays each tick (re-detected fresh from input queue)
        self.conflict_level *= (1.0 - 0.5 * dt)

    def receive_signal(self, signal: Signal):
        """
        Override to detect conflict: if two different sources send signals
        within CONFLICT_WINDOW of each other, register conflict.
        """
        super().receive_signal(signal)

        now = signal.timestamp
        self.recent_signal_sources[signal.source_id] = now

        # Check for conflicting recent sources (different region, close in time)
        for source_id, ts in self.recent_signal_sources.items():
            if source_id == signal.source_id:
                continue
            if abs(now - ts) < self.CONFLICT_WINDOW:
                conflict_signature = tuple(sorted([source_id, signal.source_id]))
                exposures = self.conflict_habituation.get(conflict_signature, 0)
                habituation_factor = max(0.1, 1.0 - (exposures * 0.05))

                self.conflict_level = min(1.0, self.conflict_level + 0.4 * habituation_factor)
                self.conflict_habituation[conflict_signature] = exposures + 1

        # Detect error: alarm signal with high threat but no PFC regulation present
        if signal.metadata.get("purpose") == "alarm":
            threat = signal.metadata.get("threat", 0.0) if hasattr(signal, "metadata") else 0.0
            if threat > 0.5 and "prefrontal" not in [
                s for s in self.recent_signal_sources if "prefrontal" in s
            ]:
                self.error_signal = min(1.0, self.error_signal + threat * 0.3)

    def get_output_signals(self) -> list[Signal]:
        strength = self.activation

        return [
            Signal(
                source_id=self.region_id,
                signal_type="glutamate",
                strength=strength * 0.8,
                metadata={"target_hint": "prefrontal", "purpose": "conflict_alarm",
                          "conflict_level": round(self.conflict_level, 4),
                          "effort": round(self.effort_output, 4)}
            ),
            Signal(
                source_id=self.region_id,
                signal_type="norepinephrine",
                strength=strength * 0.4,
                metadata={"target_hint": "hypothalamus", "purpose": "arousal_request"}
            ),
        ]

    def _get_refractory_period(self) -> float:
        return 0.05

    def get_state(self) -> dict:
        state = super().get_state()
        state.update({
            "conflict_level": round(self.conflict_level, 4),
            "effort_output": round(self.effort_output, 4),
            "error_signal": round(self.error_signal, 4),
        })
        return state