import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Signal:
    """A signal traveling between regions."""
    source_id: str
    signal_type: str        # glutamate, gaba, dopamine, serotonin, norepinephrine
    strength: float         # 0.0 to 1.0
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


@dataclass
class Connection:
    """A synapse between two regions. Grows and weakens over time."""
    source_id: str
    target_id: str
    weight: float = 0.01        # starts near zero — must be earned through co-activation
    signal_type: str = "glutamate"
    activation_history: deque = field(default_factory=lambda: deque(maxlen=100))
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)

    def strengthen(self, amount: float = 0.01):
        self.weight = min(1.0, self.weight + amount)
        self.last_active = time.time()

    def weaken(self, amount: float = 0.005):
        self.weight = max(0.0, self.weight - amount)

    def should_prune(self, threshold: float = 0.005) -> bool:
        """Returns True if connection is too weak to keep."""
        age = time.time() - self.last_active
        return self.weight < threshold and age > 30.0


class BrainRegion:
    """
    Base class for all brain regions.
    
    Each region:
    - Has an input queue (receives signals from other regions)
    - Processes signals according to its own biological rules
    - Produces output signals
    - Tracks its own activation level
    - Participates in Hebbian learning (connections grow from co-activation)
    
    Subclasses override: process(), get_output_signals()
    """

    def __init__(self, region_id: str, name: str, instance_id: int = 0):
        self.region_id = region_id
        self.name = name
        self.instance_id = instance_id

        # Internal state
        self.activation: float = 0.0           # current activation 0-1
        self.baseline_activation: float = 0.1  # resting state
        self.threshold: float = 0.3            # minimum activation to fire
        self.refractory: float = 0.0           # recovery period after firing

        # Signal queues
        self.input_queue: deque = deque(maxlen=50)
        self.output_buffer: list = []

        # Neurotransmitter sensitivity (how much this region responds to each)
        self.sensitivity: dict = {
            "glutamate":      1.0,   # excitatory
            "gaba":           1.0,   # inhibitory
            "dopamine":       0.5,   # reward modulation
            "serotonin":      0.5,   # mood modulation
            "norepinephrine": 0.5,   # arousal modulation
            "cortisol":       0.3,   # stress modulation
        }

        # Activation history for Hebbian learning
        self.activation_history: deque = deque(maxlen=200)

        # 3D position for visualization (anatomically approximate)
        self.position: dict = {"x": 0.0, "y": 0.0, "z": 0.0}

        # Metadata
        self.tick_count: int = 0
        self.last_fired: float = 0.0
        self.total_firings: int = 0

    def receive_signal(self, signal: Signal):
        """Add incoming signal to queue."""
        self.input_queue.append(signal)

    def tick(self, neuromodulators: dict, dt: float = 0.1) -> list[Signal]:
        """
        Main update loop. Called every simulation step.
        
        1. Decay current activation toward baseline
        2. Process all queued input signals
        3. Apply neuromodulator effects
        4. Fire if above threshold
        5. Record to history for Hebbian learning
        6. Return output signals
        
        Returns list of output signals to broadcast.
        """
        self.tick_count += 1

        # 1. Passive decay toward baseline
        decay_rate = 0.15 * dt
        self.activation += (self.baseline_activation - self.activation) * decay_rate

        # 2. Refractory period recovery
        if self.refractory > 0:
            self.refractory = max(0.0, self.refractory - dt)
            self.activation_history.append(self.activation)
            return []

        # 3. Process input queue
        while self.input_queue:
            signal = self.input_queue.popleft()
            self._integrate_signal(signal)

        # 4. Apply neuromodulator effects
        self._apply_neuromodulators(neuromodulators, dt)

        # 5. Region-specific processing (subclasses override this)
        self.process(neuromodulators, dt)

        # 6. Clamp activation
        self.activation = max(0.0, min(1.0, self.activation))

        # 7. Record history
        self.activation_history.append(self.activation)

        # 8. Fire if above threshold
        output_signals = []
        if self.activation >= self.threshold and self.refractory == 0:
            output_signals = self.get_output_signals()
            self.last_fired = time.time()
            self.total_firings += 1
            self.refractory = self._get_refractory_period()

        return output_signals

    def _integrate_signal(self, signal: Signal):
        """Integrate an incoming signal into current activation."""
        sensitivity = self.sensitivity.get(signal.signal_type, 0.5)
        effect = signal.strength * sensitivity

        if signal.signal_type == "gaba":
            self.activation -= effect * 0.8   # inhibitory
        else:
            self.activation += effect * 0.6   # excitatory (with diminishing returns)

    def _apply_neuromodulators(self, neuromodulators: dict, dt: float):
        """
        Global neuromodulator levels modulate how this region processes.
        Subclasses can override for region-specific effects.
        """
        dopamine = neuromodulators.get("dopamine", 0.5)
        serotonin = neuromodulators.get("serotonin", 0.5)
        norepinephrine = neuromodulators.get("norepinephrine", 0.5)

        # Norepinephrine increases baseline arousal
        arousal_boost = (norepinephrine - 0.5) * 0.1 * self.sensitivity["norepinephrine"]
        self.activation += arousal_boost * dt

    def process(self, neuromodulators: dict, dt: float):
        """
        Region-specific processing logic.
        Subclasses override this to implement biological rules.
        """
        pass

    def get_output_signals(self) -> list[Signal]:
        """
        What signals does this region emit when it fires?
        Subclasses override this.
        """
        return [Signal(
            source_id=self.region_id,
            signal_type="glutamate",
            strength=self.activation,
        )]

    def _get_refractory_period(self) -> float:
        """How long before this region can fire again. Override per region."""
        return 0.05  # 50ms default

    def get_state(self) -> dict:
        """Serialize current state for WebSocket/API transmission."""
        return {
            "region_id": self.region_id,
            "name": self.name,
            "instance_id": self.instance_id,
            "activation": round(self.activation, 4),
            "threshold": self.threshold,
            "refractory": round(self.refractory, 4),
            "total_firings": self.total_firings,
            "position": self.position,
            "tick": self.tick_count,
        }

    def __repr__(self):
        return f"<{self.name} activation={self.activation:.3f} firings={self.total_firings}>"