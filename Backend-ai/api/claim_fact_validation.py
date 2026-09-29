"""Phase 2 validation of basic claim completeness."""

from __future__ import annotations

import uuid

from api.claim_schemas import (
    FactValidationRequest,
    FactValidationResponse,
    MissingInformation,
)


class ClaimFactValidationService:
    def validate(self, request: FactValidationRequest) -> FactValidationResponse:
        missing: list[MissingInformation] = []
        claim = request.claim

        if not claim.incidents:
            missing.append(
                self._missing(
                    field_path="claim.incidents",
                    reason="At least one incident is needed to identify the event being assessed.",
                    required_for="basic_claim_assessment",
                    blocking=True,
                    question="What incident or event led to this claim?",
                )
            )
        else:
            for index, incident in enumerate(claim.incidents):
                prefix = f"claim.incidents[{index}]"
                if not incident.incident_type:
                    missing.append(
                        self._missing(
                            field_path=f"{prefix}.incident_type",
                            reason="The incident type is needed to classify the claim event.",
                            required_for="incident_classification",
                            blocking=True,
                            question="What type of incident or event occurred?",
                            related_incident_id=incident.id,
                        )
                    )
                if incident.event_date is None:
                    missing.append(
                        self._missing(
                            field_path=f"{prefix}.event_date",
                            reason="The event date is needed to determine which policy period may apply.",
                            required_for="policy_applicability",
                            blocking=True,
                            question="When did the incident or event happen?",
                            related_incident_id=incident.id,
                        )
                    )
                if not incident.location:
                    missing.append(
                        self._missing(
                            field_path=f"{prefix}.location",
                            reason="The event location is needed for territory and insured-risk checks.",
                            required_for="territory_and_risk_check",
                            blocking=True,
                            question="Where did the incident or event happen?",
                            related_incident_id=incident.id,
                        )
                    )

        facts = {fact.fact_path: fact for fact in request.facts}
        claimed_amount = facts.get("claim.claimed_amount")
        if claimed_amount is None or claimed_amount.value is None:
            missing.append(
                self._missing(
                    field_path="claim.claimed_amount",
                    reason="The claimed amount is needed for later amount assessment.",
                    required_for="amount_assessment",
                    blocking=False,
                    question="What amount is being claimed?",
                )
            )
        else:
            currency = facts.get("claim.currency")
            if currency is None or currency.value is None:
                missing.append(
                    self._missing(
                        field_path="claim.currency",
                        reason="A currency is needed to interpret the claimed amount.",
                        required_for="amount_assessment",
                        blocking=False,
                        question="What currency is the claimed amount in?",
                    )
                )

        if request.claimant_reference_required and not claim.claimant_reference:
            missing.append(
                self._missing(
                    field_path="claim.claimant_reference",
                    reason="The claimant must be identified for this type of claim.",
                    required_for="party_identification",
                    blocking=True,
                    question="Who is making or benefiting from the claim?",
                )
            )

        if request.policy_reference_required and claim.policy is None:
            missing.append(
                self._missing(
                    field_path="claim.policy",
                    reason="A policy reference is needed to identify the potentially applicable cover.",
                    required_for="policy_applicability",
                    blocking=True,
                    question="Which policy does this claim relate to?",
                )
            )

        return FactValidationResponse(
            claim_id=claim.id,
            ready_for_assessment=not any(item.blocking for item in missing),
            missing_information=missing,
        )

    @staticmethod
    def _missing(
        *,
        field_path: str,
        reason: str,
        required_for: str,
        blocking: bool,
        question: str | None = None,
        related_incident_id: uuid.UUID | None = None,
    ) -> MissingInformation:
        return MissingInformation(
            id=uuid.uuid4(),
            field_path=field_path,
            reason=reason,
            required_for=required_for,
            blocking=blocking,
            question=question,
            related_incident_id=related_incident_id,
        )
