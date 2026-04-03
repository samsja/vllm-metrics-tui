#!/usr/bin/env python3
"""Lightweight TUI for monitoring vLLM Prometheus metrics.

Usage:
    python vllm_metrics_tui.py http://localhost:8000 [http://localhost:8001 ...]
    python vllm_metrics_tui.py --help
"""

from __future__ import annotations

import argparse
import asyncio
import time
from collections import deque

import httpx
from prometheus_client.parser import text_string_to_metric_families
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, Static
from textual.widgets import Sparkline


GAUGE_METRICS = {
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
}

COUNTER_METRICS = {
    "vllm:prompt_tokens",
    "vllm:generation_tokens",
}

COUNTER_RATE_NAMES = {
    "vllm:prompt_tokens": "prompt_throughput_tps",
    "vllm:generation_tokens": "generation_throughput_tps",
}

METRIC_DISPLAY = {
    "vllm:num_requests_running": ("Requests Running", ""),
    "vllm:num_requests_waiting": ("Requests Waiting", ""),
    "vllm:kv_cache_usage_perc": ("KV Cache Usage", "%"),
    "prompt_throughput_tps": ("Prompt Throughput", "tok/s"),
    "generation_throughput_tps": ("Generation Throughput", "tok/s"),
}

POLL_INTERVAL = 5.0
WINDOW_SIZE = 60  # 5 minutes of history at 5s intervals


def parse_prometheus_text(
    text: str,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
    gauges: dict[tuple[str, str], float] = {}
    counters: dict[tuple[str, str], float] = {}
    for family in text_string_to_metric_families(text):
        if family.type == "gauge" and family.name in GAUGE_METRICS:
            for sample in family.samples:
                engine = sample.labels.get("engine", "0")
                gauges[(family.name, engine)] = sample.value
        elif family.type == "counter" and family.name in COUNTER_METRICS:
            for sample in family.samples:
                engine = sample.labels.get("engine", "0")
                counters[(family.name, engine)] = sample.value
    return gauges, counters


class MetricsStore:
    """Collects and stores metrics from multiple vLLM servers."""

    def __init__(self, urls: list[str], window_size: int = WINDOW_SIZE):
        self.urls = urls
        self.window_size = window_size
        self.history: dict[tuple[str, str, str], deque[float]] = {}
        self._prev_counters: dict[tuple[str, str, str], tuple[float, float]] = {}
        self._clients: dict[str, httpx.AsyncClient] = {}
        self._errors: dict[str, str | None] = {url: None for url in urls}

    async def start(self):
        for url in self.urls:
            self._clients[url] = httpx.AsyncClient(base_url=url, timeout=5.0)

    async def stop(self):
        for client in self._clients.values():
            await client.aclose()

    async def poll(self):
        now = time.monotonic()
        results = await asyncio.gather(
            *[self._fetch(url) for url in self.urls], return_exceptions=True
        )
        for url, result in zip(self.urls, results):
            if isinstance(result, Exception):
                self._errors[url] = str(result)
                continue
            gauges, counters = result
            self._errors[url] = None

            for (metric, engine), value in gauges.items():
                key = (url, metric, engine)
                if key not in self.history:
                    self.history[key] = deque(maxlen=self.window_size)
                self.history[key].append(value)

            for (metric, engine), value in counters.items():
                key = (url, metric, engine)
                prev = self._prev_counters.get(key)
                self._prev_counters[key] = (now, value)
                if prev is None:
                    continue
                prev_time, prev_value = prev
                dt = now - prev_time
                if dt <= 0:
                    continue
                rate = (value - prev_value) / dt
                rate_key = (url, COUNTER_RATE_NAMES[metric], engine)
                if rate_key not in self.history:
                    self.history[rate_key] = deque(maxlen=self.window_size)
                self.history[rate_key].append(rate)

    async def _fetch(
        self, url: str
    ) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], float]]:
        client = self._clients[url]
        response = await client.get("/metrics")
        response.raise_for_status()
        return parse_prometheus_text(response.text)

    def get_server_metrics(
        self, url: str
    ) -> dict[str, tuple[float, list[float]]]:
        """Returns {metric_short_name: (current_value, history_list)} for a server."""
        out: dict[str, tuple[float, list[float]]] = {}
        for (u, metric, engine), values in self.history.items():
            if u != url or not values:
                continue
            short = metric.removeprefix("vllm:")
            history_list = list(values)
            current = history_list[-1]
            out[short] = (current, history_list)
        return out

    def get_error(self, url: str) -> str | None:
        return self._errors.get(url)


class MetricPanel(Static):
    """A single metric display with sparkline."""

    DEFAULT_CSS = """
    MetricPanel {
        height: auto;
        padding: 0 1;
        margin: 0 0 1 0;
    }
    """

    def __init__(self, label: str, unit: str = "", **kwargs):
        super().__init__(**kwargs)
        self.label = label
        self.unit = unit

    def compose(self) -> ComposeResult:
        yield Static(f"{self.label}", classes="metric-label")
        yield Static("--", id=f"value-{self.id}", classes="metric-value")
        yield Sparkline([], id=f"spark-{self.id}")

    def update_metric(self, value: float, history: list[float]):
        val_widget = self.query_one(f"#value-{self.id}", Static)
        spark_widget = self.query_one(f"#spark-{self.id}", Sparkline)
        if self.unit == "%":
            val_widget.update(f"  {value:.1%}")
        elif self.unit == "tok/s":
            val_widget.update(f"  {value:.1f} {self.unit}")
        else:
            val_widget.update(f"  {value:.0f}")
        spark_widget.data = history


class ServerPanel(Static):
    """Panel for one vLLM server."""

    DEFAULT_CSS = """
    ServerPanel {
        border: solid $primary;
        height: auto;
        padding: 1;
        margin: 1;
    }

    ServerPanel.error {
        border: solid $error;
    }

    .metric-label {
        text-style: bold;
    }

    .metric-value {
        color: $success;
    }

    .server-status {
        dock: top;
        text-style: bold;
        margin-bottom: 1;
    }

    .server-error {
        color: $error;
    }

    Sparkline {
        height: 2;
        margin: 0 0 0 2;
    }
    """

    METRIC_ORDER = [
        "num_requests_running",
        "num_requests_waiting",
        "kv_cache_usage_perc",
        "prompt_throughput_tps",
        "generation_throughput_tps",
    ]

    def __init__(self, url: str, server_idx: int, **kwargs):
        super().__init__(id=f"server-{server_idx}", **kwargs)
        self.url = url
        self.server_idx = server_idx

    def compose(self) -> ComposeResult:
        yield Static(
            f"Server {self.server_idx}: {self.url}",
            classes="server-status",
            id=f"status-{self.server_idx}",
        )
        for short_name in self.METRIC_ORDER:
            full_key = f"vllm:{short_name}" if not short_name.endswith("_tps") else short_name
            label, unit = METRIC_DISPLAY.get(full_key, (short_name, ""))
            yield MetricPanel(
                label=label,
                unit=unit,
                id=f"m-{self.server_idx}-{short_name}",
            )

    def update_data(
        self, metrics: dict[str, tuple[float, list[float]]], error: str | None
    ):
        status = self.query_one(f"#status-{self.server_idx}", Static)
        if error:
            status.update(f"Server {self.server_idx}: {self.url} [red]ERROR[/red]")
            self.add_class("error")
            return
        else:
            status.update(f"Server {self.server_idx}: {self.url} [green]OK[/green]")
            self.remove_class("error")

        for short_name in self.METRIC_ORDER:
            panel_id = f"m-{self.server_idx}-{short_name}"
            panel = self.query_one(f"#{panel_id}", MetricPanel)
            if short_name in metrics:
                value, history = metrics[short_name]
                panel.update_metric(value, history)


class VllmMetricsTui(App):
    """vLLM Metrics TUI - lightweight Prometheus metrics viewer."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #servers {
        layout: grid;
        grid-gutter: 0;
    }

    #aggregate {
        dock: bottom;
        height: 3;
        padding: 0 2;
        background: $surface;
        border-top: solid $primary;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "force_refresh", "Refresh"),
    ]

    def __init__(self, urls: list[str], poll_interval: float = POLL_INTERVAL):
        super().__init__()
        self.urls = urls
        self.poll_interval = poll_interval
        self.store = MetricsStore(urls)

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id="servers"):
            for i, url in enumerate(self.urls):
                yield ServerPanel(url, i)
        yield Static("Waiting for first poll...", id="aggregate")
        yield Footer()

    async def on_mount(self):
        self.title = "vLLM Metrics"
        self.sub_title = f"Polling every {self.poll_interval:.0f}s"
        # Adjust grid columns based on server count
        servers = self.query_one("#servers")
        n = len(self.urls)
        if n <= 2:
            servers.styles.grid_size_columns = n
        elif n <= 4:
            servers.styles.grid_size_columns = 2
        else:
            servers.styles.grid_size_columns = 3

        await self.store.start()
        self.set_interval(self.poll_interval, self._poll_and_update)
        # Do an initial poll immediately
        await self._poll_and_update()

    async def _poll_and_update(self):
        await self.store.poll()
        for i, url in enumerate(self.urls):
            panel = self.query_one(f"#server-{i}", ServerPanel)
            metrics = self.store.get_server_metrics(url)
            error = self.store.get_error(url)
            panel.update_data(metrics, error)

        # Update aggregate bar
        agg = self.query_one("#aggregate", Static)
        all_metrics: dict[str, list[float]] = {}
        for url in self.urls:
            for name, (val, _) in self.store.get_server_metrics(url).items():
                all_metrics.setdefault(name, []).append(val)

        if all_metrics:
            parts = []
            for name in ServerPanel.METRIC_ORDER:
                if name not in all_metrics:
                    continue
                vals = all_metrics[name]
                full_key = f"vllm:{name}" if not name.endswith("_tps") else name
                label, unit = METRIC_DISPLAY.get(full_key, (name, ""))
                avg = sum(vals) / len(vals)
                total = sum(vals)
                if unit == "%":
                    parts.append(f"{label}: avg={avg:.1%}")
                elif unit == "tok/s":
                    parts.append(f"{label}: sum={total:.0f} tok/s")
                else:
                    parts.append(f"{label}: sum={total:.0f}")
            agg.update(" | ".join(parts))

    async def action_force_refresh(self):
        await self._poll_and_update()

    async def action_quit(self):
        await self.store.stop()
        self.exit()


def main():
    parser = argparse.ArgumentParser(
        description="Lightweight TUI for monitoring vLLM Prometheus metrics"
    )
    parser.add_argument(
        "urls",
        nargs="+",
        help="vLLM server base URLs (e.g. http://localhost:8000)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=POLL_INTERVAL,
        help=f"Poll interval in seconds (default: {POLL_INTERVAL})",
    )
    args = parser.parse_args()

    # Normalize URLs (strip trailing slash)
    urls = [url.rstrip("/") for url in args.urls]

    app = VllmMetricsTui(urls, poll_interval=args.interval)
    app.run()


if __name__ == "__main__":
    main()
