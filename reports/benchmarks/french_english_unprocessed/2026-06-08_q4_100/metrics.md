# Q4 Benchmark Metrics

| Metric | Value | Common benchmark context |
| --- | ---: | --- |
| Manifest rows | 100 | Small benchmark size |
| Scored predictions | 100 | Target: 100% coverage |
| Missing predictions | 0 | Target: 0 |
| BLEU | 31.46 | 30+ is commonly solid for MT; compare only on same dataset |
| chrF | 51.04 | 50+ is commonly solid for MT; useful with paraphrases/morphology |
| WER | 50.42% | Lower is better; secondary for MT, standard mainly for ASR |
| Source audio duration | 373.18 s | 6.22 min of source audio |
| Batch inference time | 408.6 s | Includes model load and per-file overhead |
| Source-audio real-time factor | 0.91x | Higher is faster; 1.0x means real time |