"""Tests for agentlayer.jobsched — CPU scheduling simulator."""

from __future__ import annotations

import pytest
from agentlayer.jobsched import (
    Job, Slice, run_fcfs, run_sjf, run_rr, run_priority, run_all, DEMO_JOBS,
)


# ──────────────────────────── fixtures ───────────────────────────────────────


@pytest.fixture
def simple_jobs():
    """Classic textbook example: 3 jobs arriving at t=0."""
    return [
        Job("P1", "A", burst=24, arrival=0, priority=3),
        Job("P2", "B", burst=3,  arrival=0, priority=1),
        Job("P3", "C", burst=3,  arrival=0, priority=2),
    ]


@pytest.fixture
def staggered_jobs():
    """Jobs with different arrival times."""
    return [
        Job("J1", "first",  burst=6, arrival=0, priority=2),
        Job("J2", "second", burst=2, arrival=2, priority=1),
        Job("J3", "third",  burst=8, arrival=4, priority=3),
    ]


# ──────────────────────────── FCFS ───────────────────────────────────────────


def test_fcfs_order(simple_jobs):
    res = run_fcfs(simple_jobs)
    # FCFS: same arrival, so sorted by id (P1 first)
    ids = [sl.job_id for sl in res.timeline]
    assert ids == ["P1", "P2", "P3"]


def test_fcfs_no_gaps(simple_jobs):
    res = run_fcfs(simple_jobs)
    for i in range(1, len(res.timeline)):
        assert res.timeline[i].start == res.timeline[i - 1].end


def test_fcfs_total_time(simple_jobs):
    res = run_fcfs(simple_jobs)
    assert res.total_ticks == sum(j.burst for j in simple_jobs)


def test_fcfs_turnaround(simple_jobs):
    res = run_fcfs(simple_jobs)
    # P1: 24, P2: 27, P3: 30 → avg = 27
    assert abs(res.avg_turnaround - 27.0) < 0.01


def test_fcfs_staggered(staggered_jobs):
    res = run_fcfs(staggered_jobs)
    # J1 starts at 0, J2 starts at 6, J3 starts at 8 (even though it arrives at 4)
    # — FCFS by arrival order: J1(0), J2(2), J3(4)
    assert res.timeline[0].job_id == "J1"
    assert res.timeline[0].start == 0


# ──────────────────────────── SJF ────────────────────────────────────────────


def test_sjf_picks_shortest(simple_jobs):
    res = run_sjf(simple_jobs)
    # All arrive at 0; SJF picks burst=3 first (P2 or P3), then the other 3-burst, then 24
    bursts = [next(j.burst for j in simple_jobs if j.id == sl.job_id) for sl in res.timeline]
    assert bursts == sorted(bursts)   # ascending burst order


def test_sjf_lower_avg_wait_than_fcfs(simple_jobs):
    fcfs = run_fcfs(simple_jobs)
    sjf = run_sjf(simple_jobs)
    assert sjf.avg_waiting <= fcfs.avg_waiting


def test_sjf_completes_all(staggered_jobs):
    res = run_sjf(staggered_jobs)
    scheduled_ids = {sl.job_id for sl in res.timeline}
    expected_ids = {j.id for j in staggered_jobs}
    assert scheduled_ids == expected_ids


# ──────────────────────────── Round Robin ────────────────────────────────────


def test_rr_quantum_respected():
    jobs = [Job("A", "x", burst=10, arrival=0), Job("B", "y", burst=10, arrival=0)]
    res = run_rr(jobs, quantum=3)
    for sl in res.timeline:
        assert sl.length <= 3


def test_rr_last_slice_may_be_shorter():
    jobs = [Job("A", "x", burst=7, arrival=0)]
    res = run_rr(jobs, quantum=3)
    lengths = [sl.length for sl in res.timeline]
    # A: 3, 3, 1
    assert lengths == [3, 3, 1]


def test_rr_total_time(simple_jobs):
    res = run_rr(simple_jobs, quantum=4)
    assert res.total_ticks == sum(j.burst for j in simple_jobs)


def test_rr_invalid_quantum():
    with pytest.raises(ValueError, match="quantum"):
        run_rr([Job("A", "x", burst=5)], quantum=0)


def test_rr_all_jobs_complete(staggered_jobs):
    res = run_rr(staggered_jobs, quantum=2)
    scheduled = {sl.job_id for sl in res.timeline}
    assert scheduled == {j.id for j in staggered_jobs}


# ──────────────────────────── Priority ───────────────────────────────────────


def test_priority_order(simple_jobs):
    res = run_priority(simple_jobs)
    # P2 (prio 1) < P3 (prio 2) < P1 (prio 3)
    ids = [sl.job_id for sl in res.timeline]
    assert ids == ["P2", "P3", "P1"]


def test_priority_completes_all(staggered_jobs):
    res = run_priority(staggered_jobs)
    assert {sl.job_id for sl in res.timeline} == {j.id for j in staggered_jobs}


def test_priority_tie_break_by_burst(simple_jobs):
    # Two jobs same priority, same arrival — should pick shorter burst first
    jobs = [
        Job("X", "long",  burst=10, arrival=0, priority=2),
        Job("Y", "short", burst=3,  arrival=0, priority=2),
    ]
    res = run_priority(jobs)
    assert res.timeline[0].job_id == "Y"


# ──────────────────────────── run_all ────────────────────────────────────────


def test_run_all_returns_four():
    res = run_all(DEMO_JOBS)
    assert len(res) == 4


def test_run_all_names():
    res = run_all(DEMO_JOBS)
    names = {r.algorithm for r in res}
    assert names == {"FCFS", "SJF", "Round Robin", "Priority"}


def test_run_all_same_total(simple_jobs):
    """All algorithms process the same amount of CPU work."""
    results = run_all(simple_jobs)
    expected = sum(j.burst for j in simple_jobs)
    for r in results:
        assert r.total_ticks == expected


# ──────────────────────────── Gantt chart ────────────────────────────────────


def test_gantt_non_empty():
    res = run_fcfs(DEMO_JOBS)
    chart = res.gantt_text()
    assert "█" in chart


def test_gantt_has_all_job_names():
    res = run_fcfs(DEMO_JOBS)
    chart = res.gantt_text()
    for job in DEMO_JOBS:
        assert job.name[:8] in chart


def test_gantt_ruler_present():
    res = run_fcfs(DEMO_JOBS)
    chart = res.gantt_text()
    assert "0" in chart


# ──────────────────────────── metrics ────────────────────────────────────────


def test_metrics_count():
    res = run_fcfs(DEMO_JOBS)
    assert len(res.metrics) == len(DEMO_JOBS)


def test_avg_waiting_non_negative():
    for algo_fn in (run_fcfs, run_sjf, run_priority):
        res = algo_fn(DEMO_JOBS)
        assert res.avg_waiting >= 0
    res = run_rr(DEMO_JOBS, 3)
    assert res.avg_waiting >= 0


def test_cpu_util_between_0_and_100():
    for r in run_all(DEMO_JOBS):
        assert 0.0 <= r.cpu_util <= 100.0


def test_comparison_row_has_five_fields():
    res = run_fcfs(DEMO_JOBS)
    row = res.comparison_row()
    assert len(row) == 5


# ──────────────────────────── edge cases ─────────────────────────────────────


def test_single_job_all_algos():
    jobs = [Job("only", "solo", burst=5, arrival=3, priority=1)]
    for r in run_all(jobs, quantum=2):
        assert r.total_ticks == 8  # 3 idle + 5 burst
        assert r.avg_waiting == 0.0


def test_job_burst_zero_invalid():
    with pytest.raises(ValueError):
        Job("bad", "x", burst=0)


def test_job_negative_arrival_invalid():
    with pytest.raises(ValueError):
        Job("bad", "x", burst=3, arrival=-1)


def test_large_quantum_same_as_fcfs():
    """RR with quantum ≥ max burst should behave like FCFS when all arrive at 0."""
    jobs = [Job(f"P{i}", f"job{i}", burst=i + 1, arrival=0) for i in range(4)]
    fcfs = run_fcfs(jobs)
    rr = run_rr(jobs, quantum=100)
    assert fcfs.total_ticks == rr.total_ticks
    assert abs(fcfs.avg_waiting - rr.avg_waiting) < 0.01
