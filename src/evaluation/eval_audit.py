"""Audit learned state on the exact controller used by an evaluation rollout."""
import hashlib
import json


def snapshot_agent(agent):
    rows = [(list(k), v.tolist()) for k, v in sorted(agent.q_table.items())]
    digest = hashlib.sha256(json.dumps(rows, separators=(",", ":"), allow_nan=True).encode()).hexdigest()
    return digest, agent.total_updates


def finish_eval_audit(agent, before):
    after = snapshot_agent(agent)
    return dict(q_table_sha256_before=before[0], q_table_sha256_after=after[0],
                total_updates_before=before[1], total_updates_after=after[1],
                agent_instance_used=True, invariance_pass=before == after,
                eval_mode=agent.eval_mode, epsilon=agent.epsilon,
                mutable_diagnostics=["action_counts"])
