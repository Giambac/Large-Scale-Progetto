"""
hierarchy.py — La storia dei cluster nel tempo.

Traccia di cosa è successo ai cluster durante la conversazione: quali sono stati divisi, quali sono stati uniti, da dove vengono.

Ogni volta che l'oracle fa uno split o un merge, la storia viene registrata qui.
Questo permette di navigare la gerarchia in due direzioni:
  - "drill-in": dopo uno split, scendere a vedere i due cluster figli.
  - "zoom-out": risalire al cluster genitore per vedere da dove viene un cluster.

La struttura parte vuota all'inizio e cresce solo quando l'oracle agisce — non viene pre-calcolata all'avvio.
"""
from __future__ import annotations

from dataclasses import dataclass, field

"""
class ClusterNode:
    Un nodo nell'albero dei cluster — rappresenta un singolo cluster nella storia.

    cluster_id          — l'ID del cluster.
    parent_id           — l'ID del cluster da cui proviene. None se è un cluster iniziale (creato da HDBSCAN all'avvio, senza genitore).
    children_ids        — gli ID dei cluster nati da questo cluster (dopo uno split o merge). Lista vuota se il cluster non è ancora stato diviso o unito.
    is_active           — True se il cluster esiste ancora nel clustering corrente.
                          False se è stato eliminato da uno split o merge. Anche i cluster eliminati restano nel registro per conservare la storia.
"""
@dataclass
class ClusterNode:
    cluster_id: int
    parent_id: int | None  # None = root-level cluster
    children_ids: list[int] = field(default_factory=list)
    is_active: bool = True

"""
class HierarchyStore:
    Il registro completo di tutti i cluster della sessione, attivi e non.

    Internamente è un dizionario cluster_id -> ClusterNode.
    Parte vuoto e cresce ad ogni split o merge.

    Tre operazioni disponibili:
      - register: aggiunge un nuovo cluster al registro.
      - record_split: registra che un cluster è stato diviso in due.
      - record_merge: registra che due cluster sono stati uniti in uno.
"""
@dataclass
class HierarchyStore:
    nodes: dict[int, ClusterNode] = field(default_factory=dict)

    """
    def register(self, cluster_id: int, parent_id: int | None = None) -> None:
        Aggiunge un nuovo cluster al registro.

        Viene chiamato all'inizio della sessione per ogni cluster iniziale, e poi ogni volta che uno split o merge produce un nuovo cluster.

        Crasha subito se cluster_id è già presente — un ID non viene mai riusato, quindi trovarne uno duplicato indica un bug nel sistema.
    """
    def register(self, cluster_id: int, parent_id: int | None = None) -> None:
        assert cluster_id not in self.nodes, (
            f"cluster_id {cluster_id} already registered in HierarchyStore"
        )
        self.nodes[cluster_id] = ClusterNode(
            cluster_id=cluster_id,
            parent_id=parent_id,
        )

    """
    def record_split(self, parent_id: int, child_a_id: int, child_b_id: int) -> None:
        Registra uno split: il cluster parent_id viene diviso in due nuovi cluster.

        Cosa succede:
            1. Il cluster genitore viene marcato come non più attivo.
            2. I due nuovi cluster vengono aggiunti al registro, con parent_id come genitore.
            3. Gli ID dei figli vengono salvati nel nodo del genitore.

        Crasha se parent_id non è nel registro.
    """
    def record_split(self, parent_id: int, child_a_id: int, child_b_id: int) -> None:
        assert parent_id in self.nodes, (
            f"record_split: parent_id {parent_id} not found in HierarchyStore"
        )
        self.nodes[parent_id].is_active = False
        self.nodes[parent_id].children_ids = [child_a_id, child_b_id]
        self.register(child_a_id, parent_id=parent_id)
        self.register(child_b_id, parent_id=parent_id)

    """
    def record_merge(self, parent_a_id: int, parent_b_id: int, merged_id: int) -> None:
        Registra un merge: due cluster vengono uniti in uno nuovo.

        Cosa succede:
            1. Entrambi i cluster originali vengono marcati come non più attivi.
            2. Il nuovo cluster viene aggiunto al registro. 
            Per convenzione, viene registrato come figlio di parent_a — è una scelta arbitraria per mantenere il codice deterministico.

        Crasha se uno dei due parent ID non è nel registro.
    """
    def record_merge(self, parent_a_id: int, parent_b_id: int, merged_id: int) -> None:
        assert parent_a_id in self.nodes and parent_b_id in self.nodes, (
            f"record_merge: parent_a_id {parent_a_id} or parent_b_id {parent_b_id} "
            "not found in HierarchyStore"
        )
        self.nodes[parent_a_id].is_active = False
        self.nodes[parent_b_id].is_active = False
        # Per convenzione si usa parent_a come genitore canonico del cluster risultante
        self.register(merged_id, parent_id=parent_a_id)
