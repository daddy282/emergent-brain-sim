import time
from regions.amygdala import Amygdala
from regions.hippocampus import Hippocampus
from regions.prefrontal_cortex import PrefrontalCortex
from regions.hypothalamus import Hypothalamus
from regions.vta import VTA
from regions.acc import ACC
from regions.insula import Insula
from regions.accumbens import NucleusAccumbens
from core.hebbian_engine import HebbianEngine
from regions.base_region import Signal


class BrainSimulation:
    """
    The simulation loop — brings every region to life as a single
    interconnected system.

    Responsibilities:
    1. Hold all region instances
    2. Maintain the global neuromodulator pool (dopamine, cortisol,
       norepinephrine, serotonin) — read by every region, written by
       regions that produce them (hypothalamus, VTA)
    3. Run the tick cycle: each region processes, fires, produces signals
    4. Route signals to their target_hint regions directly (direct wiring
       for known anatomical pathways)
    5. Feed all firing activity into the Hebbian engine, which forms/
       strengthens/prunes emergent connections on top of the direct wiring
    6. Expose state for visualization (3D viz reads get_state() + 
       hebbian.get_connection_graph())
    """

    def __init__(self):
        # Instantiate all regions
        self.regions: dict = {
            "amygdala_0": Amygdala(0),
            "hippocampus_0": Hippocampus(0),
            "prefrontal_0": PrefrontalCortex(0),
            "hypothalamus_0": Hypothalamus(0),
            "vta_0": VTA(0),
            "acc_0": ACC(0),
            "insula_0": Insula(0),
            "accumbens_0": NucleusAccumbens(0),
        }

        # Global neuromodulator pool — every region reads this each tick
        self.neuromodulators: dict = {
            "dopamine": 0.5,
            "serotonin": 0.5,
            "norepinephrine": 0.4,
            "cortisol": 0.3,
        }

        # Hebbian engine — builds emergent connectivity from co-activation
        self.hebbian = HebbianEngine()

        # Maps target_hint strings to region_id prefixes
        # (instance 0 for now; supports multi-instance later via _resolve_target)
        self.target_hint_map: dict = {
            "amygdala": "amygdala",
            "hippocampus": "hippocampus",
            "prefrontal": "prefrontal",
            "hypothalamus": "hypothalamus",
            "vta": "vta",
            "acc": "acc",
            "insula": "insula",
            "accumbens": "accumbens",
            "cortex": "hippocampus",  # placeholder until a cortex region exists
        }

        self.tick_count: int = 0
        self.running: bool = False

    def _resolve_target(self, target_hint: str, instance_id: int = 0) -> str:
        """Resolve a target_hint string to an actual region_id."""
        prefix = self.target_hint_map.get(target_hint)
        if prefix is None:
            return None
        return f"{prefix}_{instance_id}"

    def _route_direct_signal(self, signal: Signal):
        """Route a signal to its target_hint region, if one exists."""
        target_hint = signal.metadata.get("target_hint")
        if not target_hint:
            return

        target_id = self._resolve_target(target_hint)
        target_region = self.regions.get(target_id)
        if target_region:
            target_region.receive_signal(signal)

    def _update_neuromodulator_pool(self):
        """
        Regions that produce neuromodulators (hypothalamus, VTA) update
        the global pool. Other regions only read from it.
        """
        for region in self.regions.values():
            if hasattr(region, "get_neuromodulator_output"):
                outputs = region.get_neuromodulator_output()
                for key, value in outputs.items():
                    # Blend toward the new value rather than hard-overwrite,
                    # smooths multi-region contributions to the same modulator
                    current = self.neuromodulators.get(key, 0.5)
                    self.neuromodulators[key] = current + (value - current) * 0.3

        # Clamp all neuromodulator levels
        for key in self.neuromodulators:
            self.neuromodulators[key] = max(0.0, min(1.0, self.neuromodulators[key]))

    def tick(self, dt: float = 0.1):
        """
        Run one simulation step:
        1. Update global neuromodulator pool from producer regions
        2. Tick every region (process, fire, produce output signals)
        3. Route fired signals directly to target_hint regions
        4. Log firings into the Hebbian engine
        5. Run Hebbian tick — strengthens/forms/prunes emergent connections
           and routes additional signals through learned pathways
        """
        self.tick_count += 1

        # 1. Update neuromodulator pool from last tick's outputs
        self._update_neuromodulator_pool()

        # 2 + 3. Tick each region, route its output signals
        for region_id, region in self.regions.items():
            output_signals = region.tick(self.neuromodulators, dt)

            if output_signals:
                # 4. Log this firing for Hebbian co-activation detection
                self.hebbian.record_firing(region_id, region.activation)

                # 3. Direct anatomical routing (known pathways)
                for signal in output_signals:
                    self._route_direct_signal(signal)

        # 5. Hebbian tick — emergent connectivity on top of direct wiring
        self.hebbian.tick(self.regions, dt)

    def inject_stimulus(self, target_region_id: str, signal_type: str, strength: float, metadata: dict = None):
        """
        Inject an external stimulus into a region — this is how the
        'stimulus flood' from your blueprint enters the system.
        E.g. simulating a threatening sound hitting the amygdala.
        """
        region = self.regions.get(target_region_id)
        if not region:
            return False

        signal = Signal(
            source_id="external_stimulus",
            signal_type=signal_type,
            strength=strength,
            metadata=metadata or {}
        )
        region.receive_signal(signal)
        return True

    def get_full_state(self) -> dict:
        """
        Full snapshot for visualization / API transmission.
        Includes every region's state, the neuromodulator pool, and the
        emergent connection graph from the Hebbian engine.
        """
        return {
            "tick": self.tick_count,
            "timestamp": time.time(),
            "neuromodulators": {k: round(v, 4) for k, v in self.neuromodulators.items()},
            "regions": {rid: region.get_state() for rid, region in self.regions.items()},
            "connections": self.hebbian.get_connection_graph(),
            "hebbian_stats": self.hebbian.get_stats(),
        }

    def run(self, ticks: int = None, dt: float = 0.1, delay: float = 0.0):
        """
        Run the simulation loop. If ticks is None, runs indefinitely
        until self.running is set to False (for use in a background thread).
        """
        self.running = True
        count = 0
        while self.running:
            self.tick(dt)
            count += 1
            if ticks is not None and count >= ticks:
                break
            if delay > 0:
                time.sleep(delay)

    def stop(self):
        self.running = False