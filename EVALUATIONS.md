## Initial Evaluations of MMCTAgent

| Dataset   | Claude 3 Opus* | Claude 3 Sonnet* | Claude 3 Haiku* | GPT-4V* | Gemini 1.0 Ultra* | Gemini 1.5 Pro* | Gemini 1.0 Pro* | MMCT w/o Critic | MMCT w Critic |
|-----------|----------------|------------------|-----------------|---------|-------------------|-----------------|-----------------|-----------------|---------------|
| MMMU      | 59.40          | 53.10            | 50.20           | 56.80   | 59.40             | 58.50           | 47.90           | 59.34           | **63.57**     |
| MathVista | 50.50          | 47.90            | 46.40           | 49.90   | 53.00             | 52.10           | 45.20           | 53.30           | **56.50**     |
| MMVET     | 51.70          | 51.30            | -               | 60.20   | -                 | 64.20           | -               | 70.51           | **74.24**     |
| MMBench   | 63.30          | 67.80            | 60.70           | 77.00   | -                 | 73.60           | -               | 80.21           | **84.20**     |

## Current Evaluations
| Image Datasets | GPT-4V | MMCT with GPT-4V | GPT-4o | MMCT with GPT-4o | GPT-5 | MMCT with GPT-5 |
|----------------|--------|-------------------|--------|-------------------|--------|-------------------|
| MM-Vet         | 60.20  | 74.24             | 77.98  | 79.36             | 80.51  | 81.65             |
| MMMU           | 56.80  | 63.57             | 69.10  | 73.00             | 84.20  | 85.44             |

| Video Datasets | GPT-4o | MMCT with GPT-4o  |
|----------------|--------|-------------------|
| VideoMME       | 72.10  | 76.70             |

*Note: We are working on inclusion of LV-Bench dataset for analysis of long form videos*



