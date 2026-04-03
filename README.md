# vllm-metrics-tui

<img width="983" height="882" alt="image" src="https://github.com/user-attachments/assets/287e759f-b48a-49bb-a906-02a6ac48f3ac" />


Lightweight terminal dashboard for monitoring [vLLM](https://github.com/vllm-project/vllm) inference servers. Polls the Prometheus `/metrics` endpoint directly — no Prometheus server, no Grafana, no infrastructure needed.

Built with [Textual](https://github.com/Textualize/textual).

## Metrics

| Metric | Type | Description |
|--------|------|-------------|
| Requests Running | gauge | Number of requests currently being processed |
| Requests Waiting | gauge | Number of requests in the queue |
| KV Cache Usage | gauge | KV cache utilization percentage |
| Prompt Throughput | rate | Prompt tokens processed per second |
| Generation Throughput | rate | Tokens generated per second |

Each metric includes a sparkline showing the last 5 minutes of history.

## Install

```bash
pip install .
# or
uv pip install .
```

## Usage

```bash
# Single server
vllm-metrics-tui http://localhost:8000

# Multiple servers
vllm-metrics-tui http://gpu1:8000 http://gpu2:8000 http://gpu3:8000

# Custom poll interval (default: 5s)
vllm-metrics-tui --interval 2 http://localhost:8000

# Or run directly without installing
python vllm_metrics_tui.py http://localhost:8000
```

## Keybindings

| Key | Action |
|-----|--------|
| `q` | Quit |
| `r` | Force refresh |

## How it works

1. Polls each vLLM server's `/metrics` endpoint at a configurable interval
2. Parses the Prometheus text format using `prometheus_client.parser`
3. Computes rates from counters (prompt/generation throughput) between polls
4. Stores a sliding window of 60 samples (~5 min at default interval)
5. Renders per-server panels with current values and sparkline history
6. Aggregate stats (mean/sum across servers) shown in the bottom bar

No data is persisted — this is a live monitoring tool. For historical storage, pair with [Weights & Biases](https://wandb.ai) or a Prometheus server.
