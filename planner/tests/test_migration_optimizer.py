"""Unit tests for migration_optimizer module.

Tests cover:
- Basic cycle detection (binary, ternary, quaternary)
- Multiple independent cycles
- Repeated cycles with same pattern
- Mixed scenarios (cycles + unique migrations)
- Edge cases (empty input, non-cycle chains)
- Model swap detection (pre-fetch optimization)
"""

import pytest
from src.migration_optimizer import eliminate_redundant_migrations, detect_model_swap_pairs


# ============ Basic Cycle Tests ============

def test_binary_cycle():
    """Binary cycle: A<->B should be eliminated"""
    original = ["A", "B"]
    original_model = ["model_X", "model_Y"]
    target = ["B", "A"]
    target_model = ["model_Y", "model_X"]

    result = eliminate_redundant_migrations(original, original_model, target, target_model)

    assert len(result[0]) == 0  # All migrations cancelled
    assert len(result[4]) == 2  # Two endpoints to return to store


def test_ternary_cycle():
    """Ternary cycle: A->B->C->A should be eliminated"""
    original = ["A", "B", "C"]
    original_model = ["m1", "m2", "m3"]
    target = ["B", "C", "A"]
    target_model = ["m2", "m3", "m1"]

    result = eliminate_redundant_migrations(original, original_model, target, target_model)

    assert len(result[0]) == 0  # All migrations cancelled
    assert len(result[4]) == 3  # Three endpoints to return to store


def test_quaternary_cycle():
    """Quaternary cycle: A->B->C->D->A should be eliminated"""
    original = ["A", "B", "C", "D"]
    original_model = ["m1", "m2", "m3", "m4"]
    target = ["B", "C", "D", "A"]
    target_model = ["m2", "m3", "m4", "m1"]

    result = eliminate_redundant_migrations(original, original_model, target, target_model)

    assert len(result[0]) == 0  # All migrations cancelled


def test_mixed_cycle_and_unique():
    """Mixed scenario: A<->B redundant, C->D preserved"""
    original = ["A", "B", "C"]
    original_model = ["m1", "m2", "m3"]
    target = ["B", "A", "D"]
    target_model = ["m2", "m1", "m4"]

    result = eliminate_redundant_migrations(original, original_model, target, target_model)

    assert result[0] == ["C"]  # Only C->D preserved
    assert result[2] == ["D"]


def test_multiple_independent_cycles():
    """Multiple independent cycles: A<->B and C->D->E->C"""
    original = ["A", "B", "C", "D", "E"]
    original_model = ["m1", "m2", "m3", "m4", "m5"]
    target = ["B", "A", "D", "E", "C"]
    target_model = ["m2", "m1", "m4", "m5", "m3"]

    result = eliminate_redundant_migrations(original, original_model, target, target_model)

    assert len(result[0]) == 0  # All migrations in cycles, all eliminated


def test_non_cycle_chain():
    """Non-cycle chain A->B->C->D should be preserved"""
    original = ["A", "B", "C"]
    original_model = ["m1", "m2", "m3"]
    target = ["B", "C", "D"]  # D not in original, no cycle formed
    target_model = ["m2", "m3", "m4"]

    result = eliminate_redundant_migrations(original, original_model, target, target_model)

    assert len(result[0]) == 3  # All preserved


def test_empty_input():
    """Empty input"""
    result = eliminate_redundant_migrations([], [], [], [])
    assert result == ([], [], [], [], [])


# ============ Multiple Cycles Tests ============

def test_two_binary_cycles():
    """Two independent binary cycles: A<->B and C<->D"""
    original = ["A", "B", "C", "D"]
    original_model = ["m1", "m2", "m3", "m4"]
    target = ["B", "A", "D", "C"]
    target_model = ["m2", "m1", "m4", "m3"]

    result = eliminate_redundant_migrations(original, original_model, target, target_model)

    assert len(result[0]) == 0  # Both cycles eliminated
    assert len(result[4]) == 4  # Four endpoints to return


def test_binary_and_ternary_cycles():
    """One binary and one ternary cycle: A<->B and C->D->E->C"""
    original = ["A", "B", "C", "D", "E"]
    original_model = ["m1", "m2", "m3", "m4", "m5"]
    target = ["B", "A", "D", "E", "C"]
    target_model = ["m2", "m1", "m4", "m5", "m3"]

    result = eliminate_redundant_migrations(original, original_model, target, target_model)

    assert len(result[0]) == 0  # All cycles eliminated
    assert len(result[4]) == 5


def test_two_ternary_cycles():
    """Two independent ternary cycles: A->B->C->A and D->E->F->D"""
    original = ["A", "B", "C", "D", "E", "F"]
    original_model = ["m1", "m2", "m3", "m4", "m5", "m6"]
    target = ["B", "C", "A", "E", "F", "D"]
    target_model = ["m2", "m3", "m1", "m5", "m6", "m4"]

    result = eliminate_redundant_migrations(original, original_model, target, target_model)

    assert len(result[0]) == 0  # Both ternary cycles eliminated
    assert len(result[4]) == 6


def test_three_cycles_mixed():
    """Three mixed cycles: A<->B, C->D->E->C, F->G->H->I->F"""
    original = ["A", "B", "C", "D", "E", "F", "G", "H", "I"]
    original_model = ["m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9"]
    target = ["B", "A", "D", "E", "C", "G", "H", "I", "F"]
    target_model = ["m2", "m1", "m4", "m5", "m3", "m7", "m8", "m9", "m6"]

    result = eliminate_redundant_migrations(original, original_model, target, target_model)

    assert len(result[0]) == 0  # All three cycles eliminated
    assert len(result[4]) == 9


def test_multiple_cycles_with_remaining():
    """Multiple cycles with unique migrations: A<->B, C->D->E->C, F->G preserved"""
    original = ["A", "B", "C", "D", "E", "F"]
    original_model = ["m1", "m2", "m3", "m4", "m5", "m6"]
    target = ["B", "A", "D", "E", "C", "G"]  # G not in original
    target_model = ["m2", "m1", "m4", "m5", "m3", "m7"]

    result = eliminate_redundant_migrations(original, original_model, target, target_model)

    assert len(result[0]) == 1  # Only F->G preserved
    assert result[0] == ["F"]
    assert result[2] == ["G"]
    assert len(result[4]) == 5  # 5 endpoints from cycles to return


def test_interleaved_cycles():
    """Interleaved cycles: A<->C, B<->D (cycle members not adjacent)"""
    original = ["A", "B", "C", "D"]
    original_model = ["m1", "m2", "m3", "m4"]
    target = ["C", "D", "A", "B"]
    target_model = ["m3", "m4", "m1", "m2"]

    result = eliminate_redundant_migrations(original, original_model, target, target_model)

    assert len(result[0]) == 0  # Both binary cycles eliminated


def test_large_scale_multiple_cycles():
    """Large scale test: 4 binary cycles + 2 ternary cycles"""
    # 4 binary cycles: A<->B, C<->D, E<->F, G<->H
    # 2 ternary cycles: I->J->K->I, L->M->N->L
    original = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N"]
    original_model = [f"m{i}" for i in range(1, 15)]
    target = ["B", "A", "D", "C", "F", "E", "H", "G", "J", "K", "I", "M", "N", "L"]
    target_model = ["m2", "m1", "m4", "m3", "m6", "m5", "m8", "m7", "m10", "m11", "m9", "m13", "m14", "m12"]

    result = eliminate_redundant_migrations(original, original_model, target, target_model)

    assert len(result[0]) == 0  # All cycles eliminated
    assert len(result[4]) == 14


# ============ Repeated Cycles Tests (Same Pattern Multiple Times) ============

def test_repeated_binary_cycles_same_pattern():
    """Repeated binary cycles: multiple X<->Y cycles with same pattern (different instances)"""
    # 3 binary cycle pairs, each is model_a <-> model_b exchange
    original = ["A1", "B1", "A2", "B2", "A3", "B3"]
    original_model = ["model_a", "model_b", "model_a", "model_b", "model_a", "model_b"]
    target = ["B1", "A1", "B2", "A2", "B3", "A3"]
    target_model = ["model_b", "model_a", "model_b", "model_a", "model_b", "model_a"]

    result = eliminate_redundant_migrations(original, original_model, target, target_model)

    assert len(result[0]) == 0  # All 3 binary cycles eliminated
    assert len(result[4]) == 6


def test_repeated_ternary_cycles_same_pattern():
    """Repeated ternary cycles: multiple A->B->C->A cycles with same pattern (different instances)"""
    # 2 ternary cycles, same pattern: model_1 -> model_2 -> model_3 -> model_1
    original = ["X1", "Y1", "Z1", "X2", "Y2", "Z2"]
    original_model = ["m1", "m2", "m3", "m1", "m2", "m3"]
    target = ["Y1", "Z1", "X1", "Y2", "Z2", "X2"]
    target_model = ["m2", "m3", "m1", "m2", "m3", "m1"]

    result = eliminate_redundant_migrations(original, original_model, target, target_model)

    assert len(result[0]) == 0  # Both ternary cycles eliminated
    assert len(result[4]) == 6


def test_many_repeated_binary_cycles():
    """Many repeated binary cycles: 10 X<->Y cycles with same pattern"""
    n_pairs = 10
    original = []
    original_model = []
    target = []
    target_model = []

    for i in range(n_pairs):
        original.extend([f"A{i}", f"B{i}"])
        original_model.extend(["model_x", "model_y"])
        target.extend([f"B{i}", f"A{i}"])
        target_model.extend(["model_y", "model_x"])

    result = eliminate_redundant_migrations(original, original_model, target, target_model)

    assert len(result[0]) == 0  # All 10 binary cycles eliminated
    assert len(result[4]) == 20


def test_repeated_cycles_with_unique_migrations():
    """Repeated cycles + unique migrations: multiple repeated cycles + some non-cycle migrations"""
    # 3 repeated binary cycles + 2 unique migrations
    original = ["A1", "B1", "A2", "B2", "A3", "B3", "C", "D"]
    original_model = ["ma", "mb", "ma", "mb", "ma", "mb", "mc", "md"]
    target = ["B1", "A1", "B2", "A2", "B3", "A3", "E", "F"]  # E, F not in original
    target_model = ["mb", "ma", "mb", "ma", "mb", "ma", "me", "mf"]

    result = eliminate_redundant_migrations(original, original_model, target, target_model)

    assert len(result[0]) == 2  # Only C->E and D->F preserved
    assert set(result[0]) == {"C", "D"}
    assert set(result[2]) == {"E", "F"}
    assert len(result[4]) == 6  # 6 endpoints from cycles to return


def test_repeated_mixed_cycles():
    """Mixed repeated cycles: repeated binary + repeated ternary cycles"""
    # 2 binary cycles + 2 ternary cycles
    original = [
        "A1", "B1",  # First binary cycle
        "A2", "B2",  # Second binary cycle
        "C1", "D1", "E1",  # First ternary cycle
        "C2", "D2", "E2",  # Second ternary cycle
    ]
    original_model = [
        "ma", "mb",
        "ma", "mb",
        "mc", "md", "me",
        "mc", "md", "me",
    ]
    target = [
        "B1", "A1",
        "B2", "A2",
        "D1", "E1", "C1",
        "D2", "E2", "C2",
    ]
    target_model = [
        "mb", "ma",
        "mb", "ma",
        "md", "me", "mc",
        "md", "me", "mc",
    ]

    result = eliminate_redundant_migrations(original, original_model, target, target_model)

    assert len(result[0]) == 0  # All 4 cycles eliminated
    assert len(result[4]) == 10


def test_interleaved_repeated_cycles():
    """Interleaved repeated cycles: cycle members distributed non-adjacently"""
    # A1<->B1, A2<->B2 interleaved
    original = ["A1", "A2", "B1", "B2"]
    original_model = ["ma", "ma", "mb", "mb"]
    target = ["B1", "B2", "A1", "A2"]
    target_model = ["mb", "mb", "ma", "ma"]

    result = eliminate_redundant_migrations(original, original_model, target, target_model)

    assert len(result[0]) == 0  # Both binary cycles eliminated


# ============ Edge Cases ============

def test_single_migration_no_cycle():
    """Single migration cannot form a cycle"""
    original = ["A"]
    original_model = ["m1"]
    target = ["B"]
    target_model = ["m2"]

    result = eliminate_redundant_migrations(original, original_model, target, target_model)

    assert len(result[0]) == 1  # Single migration preserved
    assert result[0] == ["A"]


def test_self_loop_ignored():
    """Self loop (A->A) should not happen in valid input, but handle gracefully"""
    # In practice, this shouldn't happen because cur == target_model means no migration
    # But let's handle it gracefully
    original = ["A", "B"]
    original_model = ["m1", "m2"]
    target = ["A", "C"]  # A points to itself in terms of graph
    target_model = ["m1", "m3"]

    result = eliminate_redundant_migrations(original, original_model, target, target_model)

    # A->A forms a trivial cycle of length 1, should be detected
    # Actually, let's reconsider: A is in original, target[0]=A is also in original
    # So graph has edge A->A, which is a self-loop
    # Our algorithm should handle this - it will try to traverse from A,
    # and immediately find that next_node == start, forming a cycle
    assert len(result[0]) == 1  # Only B->C preserved
    assert result[0] == ["B"]


def test_partial_cycle_not_eliminated():
    """Partial cycle (broken chain) should not be eliminated"""
    # A->B->C, but C->D (not back to A)
    original = ["A", "B", "C"]
    original_model = ["m1", "m2", "m3"]
    target = ["B", "C", "D"]
    target_model = ["m2", "m3", "m4"]

    result = eliminate_redundant_migrations(original, original_model, target, target_model)

    assert len(result[0]) == 3  # All preserved (no cycle)


def test_very_long_cycle():
    """Very long cycle (10 nodes)"""
    n = 10
    original = [f"N{i}" for i in range(n)]
    original_model = [f"m{i}" for i in range(n)]
    target = [f"N{(i+1) % n}" for i in range(n)]  # Each points to next, last points to first
    target_model = [f"m{(i+1) % n}" for i in range(n)]

    result = eliminate_redundant_migrations(original, original_model, target, target_model)

    assert len(result[0]) == 0  # Long cycle eliminated
    assert len(result[4]) == n


# ============ Model Swap Detection Tests (Pre-Fetch) ============

class TestDetectModelSwapPairs:
    """Tests for detect_model_swap_pairs function (pre-fetch optimization)."""

    def test_simple_binary_swap(self):
        """Two instances swapping models should be detected."""
        endpoints = ["A", "B"]
        current_models = ["model_x", "model_y"]
        target_models = ["model_y", "model_x"]

        indices, pairs = detect_model_swap_pairs(endpoints, current_models, target_models)

        assert indices == {0, 1}
        assert len(pairs) == 1
        assert (0, 1) in pairs or (1, 0) in pairs

    def test_no_swap_different_targets(self):
        """No swap when target models don't form a cycle."""
        endpoints = ["A", "B"]
        current_models = ["model_x", "model_y"]
        target_models = ["model_z", "model_w"]  # No cycle

        indices, pairs = detect_model_swap_pairs(endpoints, current_models, target_models)

        assert indices == set()
        assert pairs == []

    def test_ternary_model_swap(self):
        """Three instances in a model rotation cycle."""
        endpoints = ["A", "B", "C"]
        current_models = ["model_x", "model_y", "model_z"]
        target_models = ["model_y", "model_z", "model_x"]

        indices, pairs = detect_model_swap_pairs(endpoints, current_models, target_models)

        assert indices == {0, 1, 2}

    def test_quaternary_model_swap(self):
        """Four instances in a model rotation cycle."""
        endpoints = ["A", "B", "C", "D"]
        current_models = ["m1", "m2", "m3", "m4"]
        target_models = ["m2", "m3", "m4", "m1"]

        indices, pairs = detect_model_swap_pairs(endpoints, current_models, target_models)

        assert indices == {0, 1, 2, 3}

    def test_mixed_swap_and_unique(self):
        """Some instances swap, others have unique migrations."""
        endpoints = ["A", "B", "C"]
        current_models = ["model_x", "model_y", "model_z"]
        target_models = ["model_y", "model_x", "model_w"]  # A<->B swap, C unique

        indices, pairs = detect_model_swap_pairs(endpoints, current_models, target_models)

        assert indices == {0, 1}
        assert 2 not in indices

    def test_two_independent_swaps(self):
        """Two independent swap pairs."""
        endpoints = ["A", "B", "C", "D"]
        current_models = ["m1", "m2", "m3", "m4"]
        target_models = ["m2", "m1", "m4", "m3"]  # A<->B and C<->D

        indices, pairs = detect_model_swap_pairs(endpoints, current_models, target_models)

        assert indices == {0, 1, 2, 3}

    def test_empty_input(self):
        """Empty input returns empty results."""
        indices, pairs = detect_model_swap_pairs([], [], [])
        assert indices == set()
        assert pairs == []

    def test_single_instance_no_swap(self):
        """Single instance cannot form a swap."""
        endpoints = ["A"]
        current_models = ["model_x"]
        target_models = ["model_y"]

        indices, pairs = detect_model_swap_pairs(endpoints, current_models, target_models)

        assert indices == set()

    def test_no_change_needed(self):
        """Instance that doesn't need change is not included in swap."""
        endpoints = ["A", "B", "C"]
        current_models = ["model_x", "model_y", "model_z"]
        target_models = ["model_y", "model_x", "model_z"]  # C doesn't need change

        indices, pairs = detect_model_swap_pairs(endpoints, current_models, target_models)

        # A<->B still detected, C not included (cur == target)
        assert indices == {0, 1}

    def test_chain_not_cycle(self):
        """Chain of model changes (no cycle back to start)."""
        endpoints = ["A", "B", "C"]
        current_models = ["m1", "m2", "m3"]
        target_models = ["m2", "m3", "m4"]  # m1->m2->m3->m4, no cycle

        indices, pairs = detect_model_swap_pairs(endpoints, current_models, target_models)

        assert indices == set()

    def test_partial_chain_with_swap(self):
        """Some instances form chain, others form swap."""
        endpoints = ["A", "B", "C", "D", "E"]
        current_models = ["m1", "m2", "m3", "m4", "m5"]
        target_models = ["m2", "m3", "m6", "m5", "m4"]  # A->m2, B->m3 chain; D<->E swap

        indices, pairs = detect_model_swap_pairs(endpoints, current_models, target_models)

        assert indices == {3, 4}  # Only D<->E detected
        assert 0 not in indices
        assert 1 not in indices

    def test_real_world_scenario(self):
        """Simulate real-world scenario from user's log."""
        # Scenario: 5 instances, some swapping models
        endpoints = [
            "http://inst1:8248",
            "http://inst2:8346",
            "http://inst3:8210",
            "http://inst4:8343",
            "http://inst5:8311"
        ]
        current_models = ["model_a", "model_b", "model_c", "model_d", "model_e"]
        # If some need to swap:
        target_models = ["model_b", "model_a", "model_d", "model_c", "model_f"]
        # inst1<->inst2 swap (a<->b), inst3<->inst4 swap (c<->d), inst5 unique

        indices, pairs = detect_model_swap_pairs(endpoints, current_models, target_models)

        assert indices == {0, 1, 2, 3}  # 4 instances in swaps
        assert 4 not in indices  # inst5 not in swap

    def test_large_rotation_cycle(self):
        """Large rotation cycle (5 instances)."""
        endpoints = [f"ep{i}" for i in range(5)]
        current_models = [f"m{i}" for i in range(5)]
        target_models = [f"m{(i+1) % 5}" for i in range(5)]  # Rotation: m0->m1->m2->m3->m4->m0

        indices, pairs = detect_model_swap_pairs(endpoints, current_models, target_models)

        assert indices == {0, 1, 2, 3, 4}

    def test_multiple_cycles_different_sizes(self):
        """Multiple cycles of different sizes."""
        # Binary: A<->B, Ternary: C->D->E->C
        endpoints = ["A", "B", "C", "D", "E"]
        current_models = ["m1", "m2", "m3", "m4", "m5"]
        target_models = ["m2", "m1", "m4", "m5", "m3"]

        indices, pairs = detect_model_swap_pairs(endpoints, current_models, target_models)

        assert indices == {0, 1, 2, 3, 4}
