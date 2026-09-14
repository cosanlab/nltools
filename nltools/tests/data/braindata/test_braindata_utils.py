"""Tests for the BrainData helpers in ``nltools.data.braindata.utils``."""

import numpy as np


class TestResolveThreshold:
    """Shared percentile-threshold resolution (#479)."""

    def test_numeric_and_none_pass_through(self):
        from nltools.data.braindata.utils import _resolve_threshold

        data = np.arange(10.0)
        assert _resolve_threshold(2.5, data) == 2.5
        assert _resolve_threshold(None, data) is None

    def test_percentile_over_finite_nonzero(self):
        from nltools.data.braindata.utils import _resolve_threshold

        # Zeros are absence-of-data (masked map) and must not skew the percentile.
        data = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, np.nan])
        expected = float(np.percentile([1.0, 2.0, 3.0, 4.0], 50))
        assert _resolve_threshold("50%", data) == expected

    def test_all_zero_data_falls_back(self):
        from nltools.data.braindata.utils import _resolve_threshold

        assert _resolve_threshold("98%", np.zeros(10)) == 0.0

    def test_bad_string_raises(self):
        with __import__("pytest").raises(ValueError, match="threshold"):
            from nltools.data.braindata.utils import _resolve_threshold

            _resolve_threshold("high", np.arange(4.0))


class TestCoalescedGC:
    """`gc.collect` is restored once every frame has left, in any exit order.

    Saving and restoring per frame is only correct for nesting. Two interleaved
    frames — enter A, enter B, exit A, exit B — had B restore the no-op A had
    installed, leaving `gc.collect` disabled for the rest of the process.
    """

    @staticmethod
    def _run_interleaved(body_a):
        """Enter A, enter B, exit A, exit B on two threads; return gc.collect."""
        import gc
        import threading

        from nltools.data.braindata.utils import _coalesced_gc

        original = gc.collect
        b_entered = threading.Event()
        a_exited = threading.Event()
        errors = []

        def thread_a():
            try:
                with _coalesced_gc():
                    b_entered.wait(5)
                    body_a()
            except _Sentinel:
                pass
            except Exception as error:  # pragma: no cover - surfaced by assert
                errors.append(error)
            finally:
                a_exited.set()

        def thread_b():
            try:
                with _coalesced_gc():
                    b_entered.set()
                    a_exited.wait(5)
            except Exception as error:  # pragma: no cover - surfaced by assert
                errors.append(error)

        threads = [threading.Thread(target=thread_a), threading.Thread(target=thread_b)]
        try:
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(10)
            assert not errors
            return gc.collect, original
        finally:
            gc.collect = original

    def test_interleaved_frames_restore_the_real_collect(self):
        current, original = self._run_interleaved(lambda: None)
        assert current is original

    def test_a_frame_that_raises_still_restores_the_real_collect(self):
        def raise_sentinel():
            raise _Sentinel()

        current, original = self._run_interleaved(raise_sentinel)
        assert current is original


class _Sentinel(Exception):
    """Raised inside a coalesced-gc frame to exercise the exception path."""
