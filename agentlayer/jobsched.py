"""CPU-scheduling simulator: FCFS, SJF, Round Robin and Priority.

Each algorithm receives the same list of :class:`Job` objects and returns a
:class:`ScheduleResult` that records the execution timeline, average waiting
time and average turnaround time.  The timeline can be rendered as an ASCII
Gantt chart or as a Rich table for the ``/sched`` REPL command.

This covers Module 3 (CPU scheduling) of the OS syllabus and demonstrates the
trade-offs between the four classic non-preemptive / time-slicing policies.

Design notes
------------
* All algorithms are **non-preemptive** except Round Robin, which preempts at
  each quantum boundary.
* SJF is non-preemptive shortest-job-*first* (not SRTF).
* Priority scheduling uses a lower number = higher priority convention.
* All times are in abstract "ticks" so the simulator is independent of wall
  clock time; real subprocesses can be slotted in by setting burst=measured_ms.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Callable


# ──────────────────────────────────────────────── data model ────────────────


@dataclass
class Job:
    """A schedulable unit of work."""

    id: str
    name: str
    burst: int      # service time in ticks (must be ≥ 1)
    arrival: int = 0  # tick at which the job becomes ready
    priority: int = 5   # lower = higher priority (for Priority scheduling)

    def __post_init__(self) -> None:
        if self.burst < 1:
            raise ValueError(f"Job {self.id!r}: burst must be ≥ 1")
        if self.arrival < 0:
            raise ValueError(f"Job {self.id!r}: arrival must be ≥ 0")


@dataclass
class Slice:
    """One continuous run of a job on the CPU."""

    job_id: str
    start: int
    end: int      # exclusive; job ran from start to end-1

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass
class JobMetrics:
    job_id: str
    name: str
    burst: int
    arrival: int
    priority: int
    finish: int = 0
    waiting: int = 0
    turnaround: int = 0


@dataclass
class ScheduleResult:
    algorithm: str
    quantum: int = 0          # only meaningful for Round Robin
    timeline: list[Slice] = field(default_factory=list)
    metrics: list[JobMetrics] = field(default_factory=list)
    avg_waiting: float = 0.0
    avg_turnaround: float = 0.0
    total_ticks: int = 0
    cpu_util: float = 0.0     # busy ticks / total ticks

    # ──────────────────────────────────── Gantt rendering ──────────────────

    def gantt_text(self, width: int = 72) -> str:
        """Return a plain-text Gantt chart that fits in ``width`` columns.

        Each tick is represented by one character; if the timeline is too wide
        the chart is scaled so the longest job's slice fits within ``width``.
        """
        if not self.timeline:
            return "(empty)"
        end = self.timeline[-1].end
        scale = max(1, (end + width - 1) // width)   # ticks per character

        # collect unique job IDs in timeline order (preserves visual order)
        seen: list[str] = []
        for sl in self.timeline:
            if sl.job_id not in seen:
                seen.append(sl.job_id)
        job_label = {m.job_id: m.name for m in self.metrics}

        rows: list[str] = []
        for jid in seen:
            label = job_label.get(jid, jid)
            bar = [" "] * ((end + scale - 1) // scale)
            for sl in self.timeline:
                if sl.job_id == jid:
                    for t in range(sl.start, sl.end):
                        col = t // scale
                        if col < len(bar):
                            bar[col] = "█"
            rows.append(f"{label[:8]:>8} │{''.join(bar)}│")

        # tick ruler
        ruler_parts = []
        t = 0
        while t <= end:
            ruler_parts.append(f"{t:<{scale}}")
            t += scale
        ruler = "".join(ruler_parts)[:width]
        rows.append(f"{'':>8}  {ruler}")
        return "\n".join(rows)

    def comparison_row(self) -> list[str]:
        """One row for the multi-algorithm comparison table."""
        q = f"  q={self.quantum}" if self.algorithm == "Round Robin" else ""
        return [
            self.algorithm + q,
            f"{self.avg_waiting:.2f}",
            f"{self.avg_turnaround:.2f}",
            f"{self.cpu_util:.1f}%",
            f"{self.total_ticks}",
        ]


# ──────────────────────────────────────────────── helpers ───────────────────


def _compute_metrics(jobs: list[Job], timeline: list[Slice]) -> tuple[list[JobMetrics], float, float]:
    """Derive per-job and aggregate metrics from a completed timeline."""
    last_end: dict[str, int] = {}
    for sl in timeline:
        last_end[sl.job_id] = max(last_end.get(sl.job_id, 0), sl.end)

    rows: list[JobMetrics] = []
    for job in jobs:
        finish = last_end.get(job.id, job.arrival)
        turnaround = finish - job.arrival
        waiting = turnaround - job.burst
        rows.append(JobMetrics(
            job_id=job.id, name=job.name, burst=job.burst,
            arrival=job.arrival, priority=job.priority,
            finish=finish, waiting=max(0, waiting), turnaround=max(0, turnaround),
        ))

    avg_w = sum(r.waiting for r in rows) / len(rows) if rows else 0.0
    avg_t = sum(r.turnaround for r in rows) / len(rows) if rows else 0.0
    return rows, avg_w, avg_t


def _make_result(algorithm: str, jobs: list[Job], timeline: list[Slice], quantum: int = 0) -> ScheduleResult:
    metrics, avg_w, avg_t = _compute_metrics(jobs, timeline)
    total = timeline[-1].end if timeline else 0
    busy = sum(sl.length for sl in timeline)
    util = busy / total * 100 if total else 0.0
    return ScheduleResult(
        algorithm=algorithm,
        quantum=quantum,
        timeline=timeline,
        metrics=metrics,
        avg_waiting=avg_w,
        avg_turnaround=avg_t,
        total_ticks=total,
        cpu_util=util,
    )


def _ready(jobs: list[Job], clock: int) -> list[Job]:
    """Jobs that have arrived by ``clock`` and are still in the queue."""
    return [j for j in jobs if j.arrival <= clock]


# ──────────────────────────────────────────── algorithms ────────────────────


def run_fcfs(jobs: list[Job]) -> ScheduleResult:
    """First-Come, First-Served — non-preemptive, sorted by arrival then id."""
    queue = sorted(copy.copy(jobs), key=lambda j: (j.arrival, j.id))
    clock = 0
    timeline: list[Slice] = []
    for job in queue:
        start = max(clock, job.arrival)
        timeline.append(Slice(job.id, start, start + job.burst))
        clock = start + job.burst
    return _make_result("FCFS", jobs, timeline)


def run_sjf(jobs: list[Job]) -> ScheduleResult:
    """Shortest-Job-First — non-preemptive; on ties, use arrival then id."""
    remaining = sorted(copy.copy(jobs), key=lambda j: (j.arrival, j.id))
    clock = 0
    timeline: list[Slice] = []
    while remaining:
        ready = _ready(remaining, clock)
        if not ready:
            clock = remaining[0].arrival   # CPU idle: jump to next arrival
            ready = _ready(remaining, clock)
        # pick shortest burst; on tie, earliest arrival, then id
        job = min(ready, key=lambda j: (j.burst, j.arrival, j.id))
        remaining.remove(job)
        start = max(clock, job.arrival)
        timeline.append(Slice(job.id, start, start + job.burst))
        clock = start + job.burst
    return _make_result("SJF", jobs, timeline)


def run_rr(jobs: list[Job], quantum: int = 2) -> ScheduleResult:
    """Round Robin — preemptive with fixed time quantum."""
    if quantum < 1:
        raise ValueError("quantum must be ≥ 1")
    # (remaining_burst, original_job)
    work: list[tuple[int, Job]] = []
    arrivals = sorted(copy.copy(jobs), key=lambda j: (j.arrival, j.id))
    clock = 0
    timeline: list[Slice] = []
    i = 0   # index into arrivals

    # seed with jobs arriving at t=0
    while i < len(arrivals) and arrivals[i].arrival <= clock:
        work.append((arrivals[i].burst, arrivals[i]))
        i += 1

    while work:
        rem, job = work.pop(0)
        run = min(rem, quantum)
        start = max(clock, job.arrival)
        clock = start + run
        timeline.append(Slice(job.id, start, clock))

        # admit new arrivals during this slice
        while i < len(arrivals) and arrivals[i].arrival <= clock:
            work.append((arrivals[i].burst, arrivals[i]))
            i += 1

        if rem - run > 0:
            work.append((rem - run, job))   # re-queue with remaining burst
        elif not work and i < len(arrivals):
            # all current work done but more jobs still to arrive
            clock = arrivals[i].arrival
            while i < len(arrivals) and arrivals[i].arrival <= clock:
                work.append((arrivals[i].burst, arrivals[i]))
                i += 1

    return _make_result("Round Robin", jobs, timeline, quantum=quantum)


def run_priority(jobs: list[Job]) -> ScheduleResult:
    """Priority scheduling — non-preemptive, lower number = higher priority."""
    remaining = sorted(copy.copy(jobs), key=lambda j: (j.arrival, j.id))
    clock = 0
    timeline: list[Slice] = []
    while remaining:
        ready = _ready(remaining, clock)
        if not ready:
            clock = remaining[0].arrival
            ready = _ready(remaining, clock)
        # highest priority (lowest number), then shortest burst, then arrival, then id
        job = min(ready, key=lambda j: (j.priority, j.burst, j.arrival, j.id))
        remaining.remove(job)
        start = max(clock, job.arrival)
        timeline.append(Slice(job.id, start, start + job.burst))
        clock = start + job.burst
    return _make_result("Priority", jobs, timeline)


# ──────────────────────────────────── convenience ────────────────────────────


def run_all(jobs: list[Job], rr_quantum: int = 2) -> list[ScheduleResult]:
    """Run all four algorithms against the same job list and return results."""
    return [
        run_fcfs(jobs),
        run_sjf(jobs),
        run_rr(jobs, rr_quantum),
        run_priority(jobs),
    ]


# ──────────────────────────────── built-in demo set ─────────────────────────

DEMO_JOBS: list[Job] = [
    Job("P1", "compiler",  burst=8,  arrival=0, priority=3),
    Job("P2", "linker",    burst=4,  arrival=1, priority=1),
    Job("P3", "test-run",  burst=9,  arrival=2, priority=4),
    Job("P4", "linter",    burst=5,  arrival=3, priority=2),
    Job("P5", "packager",  burst=2,  arrival=4, priority=5),
]
