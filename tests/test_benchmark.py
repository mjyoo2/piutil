import time
import tempfile
import json
from pathlib import Path

from piutil import Benchmark, ThroughputTracker


class TestBenchmarkBasic:
    def test_step_timing(self, capsys):
        bench = Benchmark(log_every=1, cuda_sync=False)
        bench.step_start()
        with bench.phase("work"):
            time.sleep(0.01)
        bench.step_end(0, samples=32)

        output = capsys.readouterr().out
        assert "step      0" in output
        assert "work=" in output

    def test_phase_nesting(self):
        bench = Benchmark(log_every=0, cuda_sync=False)
        bench.step_start()
        with bench.phase("forward"):
            time.sleep(0.005)
        with bench.phase("forward/transformer"):
            time.sleep(0.005)
        bench.step_end(0)

        assert "forward" in bench._phases
        assert "forward/transformer" in bench._phases

    def test_summary(self):
        bench = Benchmark(log_every=0, cuda_sync=False)
        for i in range(3):
            bench.step_start()
            with bench.phase("a"):
                time.sleep(0.005)
            with bench.phase("b"):
                time.sleep(0.01)
            bench.step_end(i, samples=8)

        s = bench.summary()
        assert "a" in s
        assert "b" in s
        assert "Count" in s
        assert "steps/sec" in s

    def test_empty_summary(self):
        bench = Benchmark(log_every=0, cuda_sync=False)
        assert bench.summary() == "No phases recorded."

    def test_extra_metrics(self, capsys):
        bench = Benchmark(log_every=1, cuda_sync=False)
        bench.step_start()
        bench.step_end(0, extra={"loss": 0.5, "lr": 1e-4})

        output = capsys.readouterr().out
        assert "loss=" in output

    def test_context_manager(self):
        with Benchmark(log_every=0, cuda_sync=False) as bench:
            bench.step_start()
            bench.step_end(0)
        # Should not raise after close


class TestBenchmarkJSONL:
    def test_jsonl_logging(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl = Path(tmpdir) / "test.jsonl"
            bench = Benchmark(jsonl_path=jsonl, log_every=0, cuda_sync=False)

            for i in range(3):
                bench.step_start()
                with bench.phase("work"):
                    time.sleep(0.005)
                bench.step_end(i, samples=16, extra={"loss": 0.1 * i})

            bench.close()

            lines = jsonl.read_text().strip().split("\n")
            assert len(lines) == 3

            record = json.loads(lines[0])
            assert record["step"] == 0
            assert "step_time" in record
            assert "time/work" in record
            assert "loss" in record

    def test_auto_jsonl_from_log_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bench = Benchmark(log_dir=tmpdir, log_every=0, cuda_sync=False)
            bench.step_start()
            bench.step_end(0)
            bench.close()

            jsonl = Path(tmpdir) / "metrics.jsonl"
            assert jsonl.exists()


class TestThroughputTracker:
    def test_basic(self):
        t = ThroughputTracker(window=5)
        for _ in range(10):
            t.record(0.1, samples=32)

        d = t.to_dict()
        assert d["throughput/steps_per_sec"] > 0
        assert d["throughput/samples_per_sec"] > 0
        assert "throughput/cumulative_steps_per_sec" in d

    def test_window(self):
        t = ThroughputTracker(window=3)
        for _ in range(10):
            t.record(0.1)

        # Only last 3 step times kept
        assert len(t._step_times) == 3

    def test_empty(self):
        t = ThroughputTracker()
        assert t.to_dict() == {}
