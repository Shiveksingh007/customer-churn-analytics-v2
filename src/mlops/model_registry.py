"""
MLflow Model Registry champion / challenger promotion gate.

Promotes a newly trained model to Production only when its ROC-AUC beats the
current Production model on the same held-out evaluation set.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PromotionDecision:
    promoted: bool
    challenger_auc: float
    production_auc: float | None
    challenger_version: str | int
    archived_versions: list[str | int]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "promoted": self.promoted,
            "challenger_auc": self.challenger_auc,
            "production_auc": self.production_auc,
            "challenger_version": self.challenger_version,
            "archived_versions": self.archived_versions,
            "reason": self.reason,
        }


def evaluate_champion_challenger(
    challenger_auc: float,
    production_auc: float | None,
    *,
    min_improvement: float = 0.0,
) -> PromotionDecision:
    """
    Pure decision logic for champion/challenger (unit-testable without MLflow).
    """
    if production_auc is None:
        return PromotionDecision(
            promoted=True,
            challenger_auc=challenger_auc,
            production_auc=None,
            challenger_version="n/a",
            archived_versions=[],
            reason="No Production model registered — promote challenger.",
        )

    if challenger_auc >= production_auc + min_improvement:
        return PromotionDecision(
            promoted=True,
            challenger_auc=challenger_auc,
            production_auc=production_auc,
            challenger_version="n/a",
            archived_versions=[],
            reason=(
                f"Challenger AUC {challenger_auc:.4f} beats Production "
                f"{production_auc:.4f} (min improvement {min_improvement:.4f})."
            ),
        )

    return PromotionDecision(
        promoted=False,
        challenger_auc=challenger_auc,
        production_auc=production_auc,
        challenger_version="n/a",
        archived_versions=[],
        reason=(
            f"Challenger AUC {challenger_auc:.4f} did not beat Production "
            f"{production_auc:.4f} — remain in Staging."
        ),
    )


def get_production_auc(client, model_name: str) -> tuple[float | None, str | None]:
    """Return (auc, version) for the current Production model version, if any."""
    versions = client.search_model_versions(f"name='{model_name}'")
    prod = [v for v in versions if v.current_stage == "Production"]
    if not prod:
        return None, None

    version = sorted(prod, key=lambda v: int(v.version), reverse=True)[0]
    run = client.get_run(version.run_id)
    auc = run.data.metrics.get("roc_auc") or run.data.metrics.get("auc")
    return (float(auc) if auc is not None else None, version.version)


def promote_model_if_better(
    model_name: str,
    challenger_run_id: str,
    challenger_auc: float,
    *,
    min_improvement: float = 0.0,
    registry_uri: str | None = None,
) -> PromotionDecision:
    """
    Register challenger run, compare AUC to Production, transition stages:
    Staging → Production (and archive previous Production) when challenger wins.
    """
    import mlflow
    from mlflow.tracking import MlflowClient

    if registry_uri:
        mlflow.set_registry_uri(registry_uri)

    client = MlflowClient()
    prod_auc, prod_version = get_production_auc(client, model_name)

    decision_core = evaluate_champion_challenger(
        challenger_auc,
        prod_auc,
        min_improvement=min_improvement,
    )

    # Register challenger artifact from run
    model_uri = f"runs:/{challenger_run_id}/model"
    mv = mlflow.register_model(model_uri, model_name)
    challenger_version = mv.version

    archived: list[str | int] = []
    if not decision_core.promoted:
        client.transition_model_version_stage(
            name=model_name,
            version=challenger_version,
            stage="Staging",
        )
        return PromotionDecision(
            promoted=False,
            challenger_auc=challenger_auc,
            production_auc=prod_auc,
            challenger_version=challenger_version,
            archived_versions=archived,
            reason=decision_core.reason,
        )

    if prod_version is not None:
        client.transition_model_version_stage(
            name=model_name,
            version=prod_version,
            stage="Archived",
        )
        archived.append(prod_version)

    client.transition_model_version_stage(
        name=model_name,
        version=challenger_version,
        stage="Production",
    )

    return PromotionDecision(
        promoted=True,
        challenger_auc=challenger_auc,
        production_auc=prod_auc,
        challenger_version=challenger_version,
        archived_versions=archived,
        reason=decision_core.reason,
    )
