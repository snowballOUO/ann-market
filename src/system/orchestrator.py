"""
Orchestrator: ties all agents together.

Per-query flow:
  1. DifficultyEstimator   -> U_t
  2. ContextCache          -> h_t
  3. PolicyAgent           -> action, propensity
  4. ExecutionAgent        -> results, L_t, C_t
  5. ShadowSampler         -> async ground-truth recall
  6. Buyer                 -> A_t, S_t
  7. ContextCache.update + LogWriter.write_async
"""
import time
from src.system.types import Query, Outcome, Trajectory
from src.system.context_cache import ContextCache


class Orchestrator:
    def __init__(
        self,
        difficulty_estimator,
        policy_agent,
        execution_agent,
        shadow_sampler,
        log_writer,
        context_cache: ContextCache = None,
    ):
        self.difficulty = difficulty_estimator
        self.policy = policy_agent
        self.execution = execution_agent
        self.shadow = shadow_sampler
        self.log = log_writer
        self.ctx = context_cache or ContextCache()

    def handle_query(self, query: Query, buyer) -> Outcome:
        t_decision = time.time()

        # 1. Difficulty
        U_t = self.difficulty.estimate(query)

        # 2. Context
        h_t = self.ctx.get_features()

        # 3. Policy decision
        action, propensity, policy_version = self.policy.decide(query, U_t, h_t)

        # 4. Execute
        results, L_t, C_t = self.execution.search(query, action.z_t)

        # 5. Shadow sample (async, non-blocking)
        if self.shadow is not None:
            self.shadow.maybe_sample(query, results)

        # 6. Buyer responds
        A_t, S_t = buyer.respond(query, results, action.p_t, L_t)

        # 7. Revenue
        R_t = (action.p_t - C_t) if A_t else (-C_t)

        outcome = Outcome(
            results=results,
            L_t=L_t,
            C_t=C_t,
            Q_t=None,           # filled in later by shadow sampler callback
            A_t=A_t,
            S_t=S_t,
            R_t=R_t,
        )

        # 8. Log + update context
        traj = Trajectory(
            query=query,
            U_t=U_t,
            h_t=h_t,
            action=action,
            propensity=propensity,
            policy_version=policy_version,
            outcome=outcome,
            timestamp=t_decision,
        )
        self.log.write_async(traj)
        self.ctx.update(outcome)

        return outcome
