import time

from piutil import get_timer, reset_timers, summary, timer


class TestTimerContextManager:
    def setup_method(self):
        reset_timers()

    def test_basic_timing(self):
        with timer("test_block"):
            time.sleep(0.05)

        stats = get_timer("test_block")
        assert stats.count == 1
        assert 0.04 < stats.total < 0.2
        assert stats.avg == stats.total

    def test_multiple_calls(self):
        for _ in range(3):
            with timer("repeated"):
                time.sleep(0.01)

        stats = get_timer("repeated")
        assert stats.count == 3
        assert stats.total > 0.03
        assert stats.min <= stats.avg <= stats.max

    def test_nested_timers(self):
        with timer("outer"):
            time.sleep(0.01)
            with timer("outer/inner"):
                time.sleep(0.01)

        outer = get_timer("outer")
        inner = get_timer("outer/inner")
        assert outer.count == 1
        assert inner.count == 1
        assert outer.total >= inner.total

    def test_zero_sleep(self):
        with timer("fast"):
            pass

        stats = get_timer("fast")
        assert stats.count == 1
        assert stats.total >= 0
        assert stats.total < 0.1


class TestTimerDecorator:
    def setup_method(self):
        reset_timers()

    def test_decorator_with_name(self):
        @timer("my_func")
        def do_work():
            time.sleep(0.01)
            return 42

        result = do_work()
        assert result == 42
        stats = get_timer("my_func")
        assert stats.count == 1

    def test_decorator_without_name(self):
        @timer
        def another_func():
            return "hello"

        result = another_func()
        assert result == "hello"
        # Should use qualname
        stats = get_timer("TestTimerDecorator.test_decorator_without_name.<locals>.another_func")
        assert stats.count == 1

    def test_decorator_preserves_args(self):
        @timer("add")
        def add(a, b, c=0):
            return a + b + c

        assert add(1, 2) == 3
        assert add(1, 2, c=10) == 13
        assert get_timer("add").count == 2


class TestTimerStats:
    def setup_method(self):
        reset_timers()

    def test_stats_properties(self):
        stats = get_timer("manual")
        stats.times = [1.0, 2.0, 3.0]

        assert stats.count == 3
        assert stats.total == 6.0
        assert stats.avg == 2.0
        assert stats.min == 1.0
        assert stats.max == 3.0
        assert stats.last == 3.0

    def test_empty_stats(self):
        stats = get_timer("empty")
        assert stats.count == 0
        assert stats.total == 0.0
        assert stats.avg == 0.0
        assert stats.min == 0.0
        assert stats.max == 0.0
        assert stats.last == 0.0

    def test_reset(self):
        with timer("to_reset"):
            pass
        assert get_timer("to_reset").count == 1

        reset_timers()
        assert get_timer("to_reset").count == 0


class TestSummary:
    def setup_method(self):
        reset_timers()

    def test_empty_summary(self):
        assert summary() == "No timers recorded."

    def test_summary_format(self):
        with timer("block_a"):
            time.sleep(0.01)
        with timer("block_b"):
            time.sleep(0.02)

        result = summary()
        assert "block_a" in result
        assert "block_b" in result
        assert "Count" in result
        assert "Total" in result

    def test_sort_by_name(self):
        with timer("zzz"):
            pass
        with timer("aaa"):
            pass

        result = summary(sort_by="name")
        lines = result.split("\n")
        data_lines = [l for l in lines if "aaa" in l or "zzz" in l]
        assert "aaa" in data_lines[0]


class TestCudaSync:
    def setup_method(self):
        reset_timers()

    def test_no_cuda_sync(self):
        """Explicitly disable cuda sync - should work without torch."""
        with timer("no_sync", cuda_sync=False):
            time.sleep(0.01)

        assert get_timer("no_sync").count == 1


class TestToDict:
    def setup_method(self):
        reset_timers()

    def test_to_dict(self):
        from piutil.timer import to_dict

        with timer("x"):
            time.sleep(0.01)

        d = to_dict()
        assert "x" in d
        assert "count" in d["x"]
        assert d["x"]["count"] == 1
        assert d["x"]["total"] > 0


class TestExceptionHandling:
    def setup_method(self):
        reset_timers()

    def test_timer_records_on_exception(self):
        """Timer should still record time even if block raises."""
        try:
            with timer("error_block"):
                time.sleep(0.01)
                raise ValueError("test error")
        except ValueError:
            pass

        stats = get_timer("error_block")
        assert stats.count == 1
        assert stats.total > 0
