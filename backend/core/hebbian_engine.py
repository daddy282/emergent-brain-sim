import time
from collections import defaultdict
from regions.base_region import BrainRegion, Connection, Signal


class HebbianEngine:
    """
    The self-building mechanism.
    
    Implements: "Neurons that fire together wire together."
    
    Every simulation tick:
    1. Check which regions fired recently
    2. If two regions co-activated within a time window, strengthen their connection
    3. If a connection hasn't been used, weaken it
    4. If a connection falls below threshold, prune it
    5. New connections can spontaneously form if co-activation is strong enough
    
    This is the engine that builds the brain's connectivity from scratch.
    No connections are hardcoded. All wiring emerges from activity.
    """

    # Hebbian learning parameters
    LEARNING_RATE = 0.008           # how fast connections strengthen
    DECAY_RATE = 0.003              # how fast unused connections weaken
    PRUNE_THRESHOLD = 0.004         # connections below this get deleted
    SPONTANEOUS_THRESHOLD = 0.6     # co-activation strength needed to form new connection
    CO_ACTIVATION_WINDOW = 0.5      # seconds — how close in time = "together"
    MAX_CONNECTIONS_PER_REGION = 50 # prevent runaway connectivity

    def __init__(self):
        # All connections: (source_id, target_id) → Connection
        self.connections: dict = {}

        # Recent firing log: region_id → timestamp
        self.recent_firings: dict = {}

        # Co-activation pairs detected this tick
        self._coactivation_candidates: list = []

        # Statistics
        self.total_connections_formed: int = 0
        self.total_connections_pruned: int = 0
        self.tick_count: int = 0

    def record_firing(self, region_id: str, activation: float):
        """Called when a region fires. Logged for co-activation detection."""
        self.recent_firings[region_id] = {
            "timestamp": time.time(),
            "activation": activation,
        }

    def tick(self, regions: dict[str, BrainRegion], dt: float = 0.1):
        """
        Main Hebbian update. Called every simulation step.
        
        1. Find all regions that fired within the co-activation window
        2. Form/strengthen connections between co-active pairs
        3. Decay all existing connections
        4. Prune dead connections
        5. Route signals through existing connections
        """
        self.tick_count += 1
        now = time.time()

        # 1. Find recently active regions
        active_regions = {
            rid: data for rid, data in self.recent_firings.items()
            if now - data["timestamp"] < self.CO_ACTIVATION_WINDOW
        }

        # 2. Hebbian strengthening — all pairs of co-active regions
        active_ids = list(active_regions.keys())
        for i in range(len(active_ids)):
            for j in range(len(active_ids)):
                if i == j:
                    continue
                source_id = active_ids[i]
                target_id = active_ids[j]

                source_act = active_regions[source_id]["activation"]
                target_act = active_regions[target_id]["activation"]
                co_activation_strength = source_act * target_act

                key = (source_id, target_id)

                if key in self.connections:
                    # Strengthen existing connection
                    hebb_delta = self.LEARNING_RATE * co_activation_strength * dt
                    self.connections[key].strengthen(hebb_delta)
                    self.connections[key].activation_history.append(co_activation_strength)

                elif co_activation_strength >= self.SPONTANEOUS_THRESHOLD:
                    # Spontaneously form a new connection
                    source_region = regions.get(source_id)
                    if source_region and self._can_add_connection(source_id):
                        new_conn = Connection(
                            source_id=source_id,
                            target_id=target_id,
                            weight=0.01,
                            signal_type=self._infer_signal_type(source_region),
                        )
                        self.connections[key] = new_conn
                        self.total_connections_formed += 1

        # 3. Decay all connections
        for key, conn in list(self.connections.items()):
            # Only decay if not recently active
            if now - conn.last_active > 1.0:
                conn.weaken(self.DECAY_RATE * dt)

        # 4. Prune dead connections
        dead = [key for key, conn in self.connections.items() if conn.should_prune()]
        for key in dead:
            del self.connections[key]
            self.total_connections_pruned += 1

        # 5. Route signals through connections
        outgoing_signals = self._route_signals(regions)

        return outgoing_signals

    def _route_signals(self, regions: dict) -> list:
        """
        For every region that fired, send weighted signals along its connections.
        The weight of the connection scales the signal strength.
        """
        routed = []
        now = time.time()

        for (source_id, target_id), conn in self.connections.items():
            source = regions.get(source_id)
            target = regions.get(target_id)

            if not source or not target:
                continue

            # Only route if source fired recently and connection is meaningful
            source_firing = self.recent_firings.get(source_id)
            if not source_firing:
                continue
            if now - source_firing["timestamp"] > self.CO_ACTIVATION_WINDOW:
                continue
            if conn.weight < self.PRUNE_THRESHOLD * 2:
                continue

            # Create weighted signal
            signal = Signal(
                source_id=source_id,
                signal_type=conn.signal_type,
                strength=source_firing["activation"] * conn.weight,
                metadata={"via_connection": True, "connection_weight": conn.weight}
            )
            target.receive_signal(signal)
            routed.append((source_id, target_id, signal.strength))

        return routed

    def _can_add_connection(self, source_id: str) -> bool:
        """Prevent any one region from becoming infinitely connected."""
        outgoing = sum(1 for (src, _) in self.connections if src == source_id)
        return outgoing < self.MAX_CONNECTIONS_PER_REGION

    def _infer_signal_type(self, region: BrainRegion) -> str:
        """
        Infer the likely neurotransmitter type from the region.
        Different regions have characteristic neurotransmitters.
        """
        region_name = region.name.lower()
        if "amygdala" in region_name:
            return "glutamate"
        if "hippocampus" in region_name:
            return "glutamate"
        if "prefrontal" in region_name:
            return "glutamate"
        if "vta" in region_name or "accumbens" in region_name:
            return "dopamine"
        if "hypothalamus" in region_name:
            return "norepinephrine"
        return "glutamate"  # default excitatory

    def get_connection_graph(self) -> list[dict]:
        """
        Serialize all connections for 3D visualization.
        Returns a list of edges with weights for Three.js rendering.
        """
        graph = []
        for (source_id, target_id), conn in self.connections.items():
            graph.append({
                "source": source_id,
                "target": target_id,
                "weight": round(conn.weight, 4),
                "signal_type": conn.signal_type,
                "age": round(time.time() - conn.created_at, 1),
                "last_active": round(time.time() - conn.last_active, 1),
            })
        return graph

    def get_stats(self) -> dict:
        return {
            "total_connections": len(self.connections),
            "total_formed": self.total_connections_formed,
            "total_pruned": self.total_connections_pruned,
            "active_regions": len(self.recent_firings),
            "tick": self.tick_count,
            "avg_weight": (
                sum(c.weight for c in self.connections.values()) / len(self.connections)
                if self.connections else 0.0
            ),
        }