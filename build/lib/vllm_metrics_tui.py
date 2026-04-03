#!/usr/bin/env python3
"""Lightweight TUI for monitoring vLLM Prometheus metrics."""

from __future__ import annotations

import argparse
import asyncio
import os
import time
from collections import deque

import httpx
import plotext as plt
from prometheus_client.parser import text_string_to_metric_families
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

POLL_INTERVAL = 5.0
WINDOW_SIZE = 120


def parse_metrics(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    engines: set[str] = set()
    for family in text_string_to_metric_families(text):
        n = family.name
        if n == "vllm:num_requests_running":
            out["running"] = sum(s.value for s in family.samples)
            for s in family.samples:
                engines.add(s.labels.get("engine", "0"))
        elif n == "vllm:num_requests_waiting":
            out["waiting"] = sum(s.value for s in family.samples)
        elif n == "vllm:kv_cache_usage_perc":
            vals = [s.value for s in family.samples]
            out["kv_cache"] = max(vals) if vals else 0
        elif n == "vllm:prompt_tokens":
            out["prompt_tokens"] = sum(s.value for s in family.samples)
        elif n == "vllm:generation_tokens":
            out["gen_tokens"] = sum(s.value for s in family.samples)
        elif n == "vllm:request_success":
            out["completed"] = sum(s.value for s in family.samples)
        elif n == "vllm:nixl_xfer_time_seconds":
            sums = [s.value for s in family.samples if s.name.endswith("_sum")]
            counts = [s.value for s in family.samples if s.name.endswith("_count")]
            out["kv_xfer_sum"] = sum(sums)
            out["kv_xfer_count"] = sum(counts)
    out["n_engines"] = len(engines) if engines else 1
    return out


class Store:
    def __init__(self, urls: list[str]):
        self.urls = urls
        self._clients: dict[str, httpx.AsyncClient] = {}
        self._raw: dict[str, dict[str, float]] = {}
        self._prev: dict[str, dict[str, float]] = {}
        self._prev_time: float = 0
        self.history: dict[str, deque[float]] = {}
        self.current: dict[str, float] = {}
        self.errors: int = 0
        self._roles: dict[str, str] = {}
        for url in urls:
            port = int(url.rsplit(":", 1)[-1].split("/")[0])
            self._roles[url] = "decode" if port >= 8200 else "prefill"
        self.n_prefill_nodes = sum(1 for r in self._roles.values() if r == "prefill")
        self.n_decode_nodes = sum(1 for r in self._roles.values() if r == "decode")

    async def start(self):
        for url in self.urls:
            self._clients[url] = httpx.AsyncClient(base_url=url, timeout=5.0)

    async def stop(self):
        for c in self._clients.values():
            await c.aclose()

    async def poll(self):
        now = time.monotonic()
        results = await asyncio.gather(
            *[self._fetch(u) for u in self.urls], return_exceptions=True
        )

        agg: dict[str, float] = {}
        self.errors = 0
        for url, res in zip(self.urls, results):
            if isinstance(res, Exception):
                self.errors += 1
                continue
            self._raw[url] = res
            for k, v in res.items():
                agg[k] = agg.get(k, 0) + v

        out: dict[str, float] = {}
        out["running"] = agg.get("running", 0)
        out["waiting"] = agg.get("waiting", 0)
        out["n_engines"] = agg.get("n_engines", 0)
        kv_vals = [r.get("kv_cache", 0) for r in self._raw.values()]
        out["kv_max"] = max(kv_vals) if kv_vals else 0

        dt = now - self._prev_time if self._prev_time else 0
        if dt > 0 and self._prev:
            prev_agg: dict[str, float] = {}
            for r in self._prev.values():
                for k, v in r.items():
                    prev_agg[k] = prev_agg.get(k, 0) + v

            def rate(key: str) -> float:
                return max((agg.get(key, 0) - prev_agg.get(key, 0)) / dt, 0)

            out["prefill_tps"] = rate("prompt_tokens")
            out["decode_tps"] = rate("gen_tokens")
            out["req_per_s"] = rate("completed")

            out["prefill_tps_node"] = out["prefill_tps"] / self.n_prefill_nodes if self.n_prefill_nodes else 0
            out["decode_tps_node"] = out["decode_tps"] / self.n_decode_nodes if self.n_decode_nodes else 0

            xd = agg.get("kv_xfer_sum", 0) - prev_agg.get("kv_xfer_sum", 0)
            xc = agg.get("kv_xfer_count", 0) - prev_agg.get("kv_xfer_count", 0)
            out["kv_xfer_ms"] = (xd / xc * 1000) if xc > 0 else 0

        self._prev = {u: dict(r) for u, r in self._raw.items()}
        self._prev_time = now

        for k, v in out.items():
            self.history.setdefault(k, deque(maxlen=WINDOW_SIZE)).append(v)

        self.current = out

    async def _fetch(self, url: str) -> dict[str, float]:
        resp = await self._clients[url].get("/metrics")
        resp.raise_for_status()
        return parse_metrics(resp.text)


def make_graph(data: list[float], width: int, height: int, title: str, color: str = "cyan", y_label: str = "") -> str:
    plt.clear_figure()
    plt.theme("clear")
    plt.plotsize(width, height)
    if data:
        plt.plot(list(range(len(data))), data, color=color)
        if len(data) > 1:
            plt.xlim(0, WINDOW_SIZE)
    else:
        plt.plot([0], [0], color=color)
    plt.xaxes(False)
    plt.yaxes(True, True)
    plt.title(title)
    return plt.build()


def build_dashboard(store: Store, term_width: int, term_height: int) -> Layout:
    c = store.current
    n_ok = len(store.urls) - store.errors
    n_total = len(store.urls)

    # Status line
    if store.errors == 0:
        status = Text(f" {n_ok}/{n_total} nodes healthy", style="bold green")
    else:
        status = Text(f" {n_ok}/{n_total} nodes healthy ({store.errors} errors)", style="bold red")

    # Summary table
    tbl = Table(show_header=False, expand=True, box=None, padding=(0, 2))
    tbl.add_column("metric", style="bold white", width=16)
    tbl.add_column("value", style="bold cyan", justify="right", width=14)
    tbl.add_column("metric", style="bold white", width=16)
    tbl.add_column("value", style="bold cyan", justify="right", width=14)

    def fv(key: str, unit: str) -> str:
        v = c.get(key)
        if v is None:
            return "[dim]--[/dim]"
        if unit == "%":
            return f"[bold cyan]{v:.1%}[/bold cyan]"
        if unit in ("tok/s", "req/s", "ms"):
            return f"[bold cyan]{v:,.1f}[/bold cyan] [dim]{unit}[/dim]"
        return f"[bold cyan]{v:,.0f}[/bold cyan]"

    n_gpus = int(c.get("n_engines", 0)) if "n_engines" in c else 0
    gpu_label = f" [dim]({n_gpus} GPUs)[/dim]" if n_gpus else ""

    tbl.add_row("Prefill", fv("prefill_tps", "tok/s"), "Decode", fv("decode_tps", "tok/s"))
    tbl.add_row("Prefill/node", fv("prefill_tps_node", "tok/s"), "Decode/node", fv("decode_tps_node", "tok/s"))
    tbl.add_row("Running", fv("running", ""), "Waiting", fv("waiting", ""))
    tbl.add_row("Completed", fv("req_per_s", "req/s"), "KV Transfer", fv("kv_xfer_ms", "ms"))
    tbl.add_row("KV Cache (max)", fv("kv_max", "%"), "Nodes", f"[bold cyan]{store.n_prefill_nodes}P + {store.n_decode_nodes}D[/bold cyan]")

    summary = Panel(
        Group(status, tbl),
        title="[bold]vLLM Inference[/bold]",
        border_style="blue",
    )

    # Graphs - 2 columns, 3 rows
    graph_w = max((term_width // 2) - 4, 20)
    graph_h = max((term_height - 14) // 3, 5)

    graphs = [
        ("prefill_tps", "Prefill tok/s", "cyan"),
        ("decode_tps", "Decode tok/s", "green"),
        ("running", "Running Requests", "yellow"),
        ("waiting", "Waiting Requests", "red"),
        ("kv_max", "KV Cache (max)", "magenta"),
        ("kv_xfer_ms", "KV Transfer (ms)", "blue"),
    ]

    layout = Layout()
    layout.split_column(
        Layout(summary, name="summary", size=10),
        Layout(name="graphs"),
    )

    graph_rows = []
    for i in range(0, len(graphs), 2):
        row = Layout(name=f"grow{i}")
        left_key, left_title, left_color = graphs[i]
        left_data = list(store.history.get(left_key, []))
        left_graph = make_graph(left_data, graph_w, graph_h, left_title, left_color)

        if i + 1 < len(graphs):
            right_key, right_title, right_color = graphs[i + 1]
            right_data = list(store.history.get(right_key, []))
            right_graph = make_graph(right_data, graph_w, graph_h, right_title, right_color)
            row.split_row(
                Layout(Panel(Text.from_ansi(left_graph), border_style="dim"), name=f"g{i}"),
                Layout(Panel(Text.from_ansi(right_graph), border_style="dim"), name=f"g{i+1}"),
            )
        else:
            row.split_row(
                Layout(Panel(Text.from_ansi(left_graph), border_style="dim"), name=f"g{i}"),
            )
        graph_rows.append(row)

    layout["graphs"].split_column(*graph_rows)
    return layout


async def run(urls: list[str], interval: float):
    store = Store(urls)
    await store.start()

    console = Console()
    try:
        with Live(console=console, refresh_per_second=1, screen=True) as live:
            while True:
                await store.poll()
                w = console.width
                h = console.height
                layout = build_dashboard(store, w, h)
                live.update(layout)
                await asyncio.sleep(interval)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await store.stop()


def main():
    parser = argparse.ArgumentParser(description="vLLM Metrics TUI")
    parser.add_argument("urls", nargs="+", help="vLLM server URLs")
    parser.add_argument("--interval", type=float, default=POLL_INTERVAL)
    args = parser.parse_args()
    asyncio.run(run([u.rstrip("/") for u in args.urls], args.interval))


if __name__ == "__main__":
    main()
