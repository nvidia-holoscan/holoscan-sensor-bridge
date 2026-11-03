## Description: <br>
Create or port Hololink example apps across receiver types, cameras, and pipelines. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and engineers working with Holoscan Sensor Bridge hardware who need to create or port Hololink example applications across different receiver types, cameras, and processing pipelines. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [No] <br>
**Credential Type(s):** [None] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Receiver Reference](references/receiver-reference.md) <br>
- [Pipeline Reference](references/pipeline-reference.md) <br>
- [Worked Example](references/worked-example.md) <br>


## Skill Output: <br>
**Output Type(s):** [Code, Files] <br>
**Output Format:** [Python source files] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-4-8`) <br>
- Codex (`openai/openai/gpt-5.5`) <br>



## Evaluation Tasks: <br>
11 evaluation tasks (11 positive), each run in an isolated sandbox pod. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Whether the skill avoids unsafe operations, secret leakage, and unauthorized access. <br>
- Correctness: Whether the final answer is correct against the reference answer. <br>
- Discoverability: Whether the right skill was loaded and executed when needed. <br>
- Effectiveness: Whether the skill helped the agent complete the user's goal and follow the expected workflow. <br>
- Efficiency: Whether the skill avoided wasted tool or skill usage. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Whether the expected skill was found and executed. <br>
- `skill_efficiency`: Routing quality, workspace-aware skill reads, and productive tool use. <br>
- `accuracy`: Final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Whether the user's goal was achieved. <br>
- `behavior_check`: Whether the expected workflow behavior was followed. <br>



## Evaluation Results: <br>
| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 36% → 82% (+46 points) | 35% → 69% (+34 points) |
| Security | 86% → 100% (+14 points) | 59% → 45% (-14 points) |
| Correctness | 15% → 80% (+65 points) | 35% → 84% (+49 points) |
| Discoverability | 48% → 99% (+51 points) | 32% → 73% (+41 points) |
| Effectiveness | 5% → 58% (+52 points) | 14% → 57% (+42 points) |
| Efficiency | 26% → 72% (+46 points) | 38% → 87% (+50 points) |

## Skill Version(s): <br>
1.0.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
