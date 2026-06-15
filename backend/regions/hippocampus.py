import time
from collections import deque
from .base_region import BrainRegion, Signal


class Hippocampus(BrainRegion):
    """
    The hippocampus — memory encoding, pattern completion, novelty detection.
    
    Biological rules implemented:
    - Novelty detection: fires strongly to new patterns, habituates to repeated ones
    - Emotional tagging: amygdala signals boost memory encoding strength
    - Pattern completion: partial input activates full stored pattern
    - Sleep consolidation: during offline phase, replays and transfers to cortex
    - Stress impairment: high cortisol chronically → hippocampal damage simulation
    - Spatial/contextual mapping: tracks context of each experience
    """

    def __init__(self, instance_id: int = 0):
        super().__init__(
            region_id=f"hippocampus_{instance_id}",
            name="Hippocampus",
            instance_id=instance_id
        )

        # Memory stores
        self.episodic_memory: list = []            # list of encoded episodes
        self.pattern_store: dict = {}              # pattern_id → activation vector
        self.working_context: dict = {}            # current context being encoded

        # Novelty tracking
        self.novelty_score: float = 0.0           # how novel is current input
        self.habituation_map: dict = {}            # stimulus_hash → exposure_count

        # Emotional tagging from amygdala
        self.emotional_tag_strength: float = 0.0  # current emotional weight
        self.pending_emotional_tag: float = 0.0   # incoming from amygdala

        # Consolidation
        self.consolidation_buffer: deque = deque(maxlen=50)  # recent experiences
        self.sleep_mode: bool = False              # during offline consolidation
        self.replay_index: int = 0

        # Stress damage accumulation
        self.stress_damage: float = 0.0           # 0-1, reduces encoding efficiency

        # Anatomical position
        self.position = {"x": -0.25, "y": -0.18, "z": -0.08}

        self.threshold = 0.25
        self.baseline_activation = 0.12
        self.sensitivity["cortisol"] = 0.9        # very sensitive to stress
        self.sensitivity["dopamine"] = 0.6        # dopamine gates encoding

    def process(self, neuromodulators: dict, dt: float):
        cortisol = neuromodulators.get("cortisol", 0.3)
        dopamine = neuromodulators.get("dopamine", 0.5)
        norepinephrine = neuromodulators.get("norepinephrine", 0.4)

        # 1. Chronic stress impairs encoding
        if cortisol > 0.7:
            self.stress_damage = min(1.0, self.stress_damage + 0.0005 * dt)
            self.activation -= cortisol * 0.1 * dt  # stress suppresses hippocampus

        # 2. Apply emotional tag — amygdala-flagged events encode more strongly
        if self.pending_emotional_tag > 0:
            encoding_boost = self.pending_emotional_tag * 0.5
            self.activation += encoding_boost * dt
            self.emotional_tag_strength = self.pending_emotional_tag
            self.pending_emotional_tag = 0.0

        # 3. Novelty detection — fire more for novel inputs, less for familiar ones
        novelty_boost = self.novelty_score * 0.4
        self.activation += novelty_boost * dt

        # 4. Dopamine gates memory encoding — optimal at moderate levels
        # (Yerkes-Dodson: too low = no encoding, too high = encoding degrades)
        dopamine_gate = 1.0 - abs(dopamine - 0.55) * 1.5
        dopamine_gate = max(0.1, dopamine_gate)
        self.activation *= (0.7 + 0.3 * dopamine_gate)

        # 5. Sleep consolidation replay
        if self.sleep_mode:
            self._replay_consolidation(dt)

    def receive_emotional_tag(self, strength: float, valence: float):
        """Called when amygdala sends emotional tagging signal."""
        self.pending_emotional_tag = strength
        if self.working_context:
            self.working_context["emotional_weight"] = strength
            self.working_context["valence"] = valence

    def encode_experience(self, stimulus_id: str, activation_pattern: dict):
        """
        Encode a new experience into episodic memory.
        Strength of encoding depends on: novelty, emotional weight, stress level.
        """
        efficiency = max(0.1, 1.0 - self.stress_damage)
        encoding_strength = (
            self.novelty_score * 0.4 +
            self.emotional_tag_strength * 0.4 +
            self.activation * 0.2
        ) * efficiency

        if encoding_strength > 0.1:
            episode = {
                "id": stimulus_id,
                "timestamp": time.time(),
                "pattern": activation_pattern.copy(),
                "strength": encoding_strength,
                "emotional_weight": self.emotional_tag_strength,
                "context": self.working_context.copy(),
                "novelty": self.novelty_score,
            }
            self.episodic_memory.append(episode)
            self.consolidation_buffer.append(episode)

            # Update habituation
            self.habituation_map[stimulus_id] = \
                self.habituation_map.get(stimulus_id, 0) + 1
            # Reduce novelty for repeated stimuli
            exposures = self.habituation_map[stimulus_id]
            self.novelty_score = max(0.0, 1.0 - (exposures / 10.0))

        return encoding_strength

    def pattern_complete(self, partial_input: dict, threshold: float = 0.3) -> dict:
        """
        Given a partial pattern, return the best matching stored memory.
        This is pattern completion — how memories are retrieved from partial cues.
        """
        best_match = None
        best_score = threshold

        for episode in self.episodic_memory[-100:]:  # search recent memories
            score = self._pattern_similarity(partial_input, episode["pattern"])
            if score > best_score:
                best_score = score
                best_match = episode

        if best_match:
            # Retrieval activates the hippocampus
            self.activation = min(1.0, self.activation + best_score * 0.3)

        return best_match

    def _pattern_similarity(self, a: dict, b: dict) -> float:
        """Simple cosine-like similarity between two activation patterns."""
        shared_keys = set(a.keys()) & set(b.keys())
        if not shared_keys:
            return 0.0
        dot = sum(a[k] * b[k] for k in shared_keys)
        mag_a = sum(v**2 for v in a.values()) ** 0.5
        mag_b = sum(v**2 for v in b.values()) ** 0.5
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    def enter_sleep_mode(self):
        """Start offline consolidation phase."""
        self.sleep_mode = True
        self.replay_index = 0

    def exit_sleep_mode(self):
        """End consolidation phase."""
        self.sleep_mode = False
        self.consolidation_buffer.clear()

    def _replay_consolidation(self, dt: float):
        """Replay recent experiences to consolidate to cortex."""
        if not self.consolidation_buffer:
            self.sleep_mode = False
            return

        memories = list(self.consolidation_buffer)
        if self.replay_index < len(memories):
            memory = memories[self.replay_index]
            # Replay activates hippocampus
            self.activation = min(1.0, memory["strength"] * 0.8)
            self.replay_index += 1
        else:
            self.sleep_mode = False

    def get_output_signals(self) -> list[Signal]:
        """
        Hippocampus outputs:
        - To cortex: memory consolidation signals
        - To amygdala: contextual information about current situation
        - To prefrontal: episodic context for decision making
        """
        strength = self.activation

        signals = [
            Signal(
                source_id=self.region_id,
                signal_type="glutamate",
                strength=strength * 0.7,
                metadata={"target_hint": "prefrontal", "purpose": "episodic_context",
                          "memory_count": len(self.episodic_memory),
                          "novelty": self.novelty_score}
            ),
            Signal(
                source_id=self.region_id,
                signal_type="glutamate",
                strength=strength * 0.5,
                metadata={"target_hint": "amygdala", "purpose": "context_retrieval"}
            ),
        ]

        # During sleep consolidation, strong signal to cortex
        if self.sleep_mode:
            signals.append(Signal(
                source_id=self.region_id,
                signal_type="glutamate",
                strength=strength,
                metadata={"target_hint": "cortex", "purpose": "consolidation",
                          "replay_index": self.replay_index}
            ))

        return signals

    def get_state(self) -> dict:
        state = super().get_state()
        state.update({
            "memory_count": len(self.episodic_memory),
            "novelty_score": round(self.novelty_score, 4),
            "emotional_tag_strength": round(self.emotional_tag_strength, 4),
            "stress_damage": round(self.stress_damage, 4),
            "sleep_mode": self.sleep_mode,
            "consolidation_buffer_size": len(self.consolidation_buffer),
        })
        return state