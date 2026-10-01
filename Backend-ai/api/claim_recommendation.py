"""Phase 7 deterministic aggregation into a decision-support recommendation."""

from __future__ import annotations

import uuid

from api.claim_schemas import (
    CalculationResult,
    ClaimRecommendation,
    ClaimRecommendationRequest,
    ClaimRecommendationResponse,
    CoverageAssessment,
)


DISCLAIMER = (
    "Decision-support recommendation only; the insurer must make the final claim decision."
)


class ClaimRecommendationService:
    def recommend(self, request: ClaimRecommendationRequest) -> ClaimRecommendationResponse:
        recommendation = self._recommend(request)
        return ClaimRecommendationResponse(claim_id=request.claim.id, recommendation=recommendation)

    def _recommend(self, request: ClaimRecommendationRequest) -> ClaimRecommendation:
        coverages = request.coverage_assessments
        if not coverages:
            return self._result(
                "no_recommendation",
                ["No coverage assessment is available to support a recommendation."],
            )

        unavailable = [
            item for item in request.missing_information
            if item.required_for == "obligation_assessment"
            and item.metadata.get("assessment_unavailable") is True
        ]
        if unavailable:
            return self._result(
                "needs_human_review",
                ["The obligation assessment is unavailable and requires human review."],
                human_review=True,
            )

        exclusions = self._group(request.exclusion_assessments)
        obligations = self._group(request.obligation_assessments)
        calculations = self._group(request.calculation_results)
        declined = {
            coverage.id: self._decline_basis(coverage, exclusions.get(coverage.id, []))
            for coverage in coverages
        }

        if all(declined[coverage.id] for coverage in coverages):
            reasons: list[str] = []
            supporting: list[uuid.UUID] = []
            for coverage in coverages:
                basis_reasons, basis_ids = declined[coverage.id]
                reasons.extend(basis_reasons)
                supporting.extend(basis_ids)
            return self._result("recommend_decline", reasons, supporting)

        active = [coverage for coverage in coverages if not declined[coverage.id]]
        blocking = self._relevant_blocking_missing(request, active)
        if blocking:
            fields = ", ".join(dict.fromkeys(item.field_path for item in blocking))
            supporting = self._ids_with_missing(active, exclusions, obligations, calculations)
            return self._result(
                "needs_information",
                [f"Blocking information is still required: {fields}."],
                supporting,
            )

        incomplete_assessments = self._missing_dependent_assessments(
            active, exclusions, obligations
        )
        if incomplete_assessments:
            return self._result(
                "needs_information",
                ["Required facts are missing for one or more structured assessments."],
                incomplete_assessments,
            )

        if any(declined[coverage.id] for coverage in coverages):
            ids = []
            for coverage in coverages:
                ids.append(coverage.id)
                ids.extend(declined[coverage.id][1] if declined[coverage.id] else [])
            return self._result(
                "needs_human_review",
                ["Different coverage assessments produce conflicting support and decline outcomes."],
                ids,
                human_review=True,
            )

        ambiguous = self._ambiguous_assessments(active, exclusions, obligations)
        if ambiguous:
            return self._result(
                "needs_human_review",
                ["One or more structured assessments remain ambiguous without a missing-fact resolution."],
                ambiguous,
                human_review=True,
            )

        breached = [
            obligation
            for coverage in active
            for obligation in obligations.get(coverage.id, [])
            if obligation.status == "breached"
        ]
        if breached:
            consequence = any(item.permitted_consequence for item in breached)
            reason = (
                "A breached obligation has a supported consequence that requires human interpretation."
                if consequence else
                "An obligation breach is established, but no deterministic claim outcome is supported."
            )
            return self._result(
                "needs_human_review", [reason], [item.id for item in breached], human_review=True
            )

        selected_calculations: list[CalculationResult] = []
        for coverage in active:
            coverage_calculations = calculations.get(coverage.id, [])
            if not coverage_calculations or any(item.status == "incomplete" for item in coverage_calculations):
                ids = [coverage.id, *(item.id for item in coverage_calculations)]
                return self._result(
                    "needs_information",
                    [f"A complete payable-amount calculation is required for {coverage.coverage_reference}."],
                    ids,
                )
            completed = [item for item in coverage_calculations if item.status == "complete"]
            if len(completed) != 1 or len(coverage_calculations) != 1:
                return self._result(
                    "needs_human_review",
                    [f"Calculation results conflict for {coverage.coverage_reference}."],
                    [coverage.id, *(item.id for item in coverage_calculations)],
                    human_review=True,
                )
            selected_calculations.append(completed[0])

        currencies = {item.currency for item in selected_calculations if item.currency}
        if len(currencies) != 1 or any(item.currency is None for item in selected_calculations):
            return self._result(
                "needs_human_review",
                ["Completed calculations do not provide one consistent currency."],
                [item.id for item in selected_calculations],
                human_review=True,
            )
        if any(item.payable_amount is None or item.payable_amount <= 0 for item in selected_calculations):
            return self._result(
                "needs_human_review",
                ["A completed calculation does not contain a positive payable amount."],
                [item.id for item in selected_calculations],
                human_review=True,
            )

        if len(selected_calculations) != 1:
            return self._result(
                "needs_human_review",
                ["Multiple payable amounts require an aggregation rule outside Phase 7."],
                [item.id for item in selected_calculations],
                human_review=True,
            )

        payable = selected_calculations[0].payable_amount
        reduced = []
        inconsistent = []
        for item in selected_calculations:
            reference = item.eligible_amount if item.eligible_amount is not None else item.claimed_amount
            if reference is None or item.payable_amount > reference:
                inconsistent.append(item)
            elif item.payable_amount < reference:
                if item.calculation_steps:
                    reduced.append(item)
                else:
                    inconsistent.append(item)
        if inconsistent:
            return self._result(
                "needs_human_review",
                ["The completed payable amount cannot be reconciled with its claimed or eligible amount."],
                [item.id for item in inconsistent],
                human_review=True,
            )

        supporting = [coverage.id for coverage in active]
        for coverage in active:
            supporting.extend(item.id for item in exclusions.get(coverage.id, [])
                              if item.status == "does_not_apply")
            supporting.extend(item.id for item in obligations.get(coverage.id, [])
                              if item.status == "satisfied")
        supporting.extend(item.id for item in selected_calculations)
        currency = next(iter(currencies))
        if reduced:
            return self._result(
                "recommend_partial",
                ["Completed deterministic calculations apply supported reductions to the eligible amount."],
                supporting,
                amount=payable,
                currency=currency,
            )
        return self._result(
            "recommend_approve",
            ["Coverage is confirmed and completed calculations support a positive payable amount."],
            supporting,
            amount=payable,
            currency=currency,
        )

    @staticmethod
    def _group(items):
        grouped = {}
        for item in items:
            grouped.setdefault(item.coverage_assessment_id, []).append(item)
        return grouped

    @staticmethod
    def _decline_basis(coverage: CoverageAssessment, exclusions):
        if coverage.status == "not_covered":
            return ([f"Coverage {coverage.coverage_reference} is definitively not covered."], [coverage.id])
        applying = [item for item in exclusions if item.status == "applies"]
        if applying:
            return (
                [f"A confirmed exclusion applies to coverage {coverage.coverage_reference}."],
                [coverage.id, *(item.id for item in applying)],
            )
        return None

    @staticmethod
    def _relevant_blocking_missing(request, active):
        active_exposures = {item.exposure_id for item in active if item.exposure_id is not None}
        has_unbound_active = any(item.exposure_id is None for item in active)
        return [
            item for item in request.missing_information
            if item.blocking and (
                item.related_exposure_id is None
                or item.related_exposure_id in active_exposures
                or has_unbound_active
            )
        ]

    @staticmethod
    def _ids_with_missing(active, exclusions, obligations, calculations):
        ids = []
        for coverage in active:
            ids.append(coverage.id)
            ids.extend(item.id for item in exclusions.get(coverage.id, []) if item.missing_information_ids)
            ids.extend(item.id for item in obligations.get(coverage.id, []) if item.missing_information_ids)
            ids.extend(item.id for item in calculations.get(coverage.id, []) if item.missing_information_ids)
        return ids

    @staticmethod
    def _missing_dependent_assessments(active, exclusions, obligations):
        ids = []
        for coverage in active:
            if (coverage.status in {"potentially_covered", "indeterminate"}
                    and coverage.missing_information_ids):
                ids.append(coverage.id)
            ids.extend(item.id for item in exclusions.get(coverage.id, [])
                       if item.status == "indeterminate" and item.missing_information_ids)
            ids.extend(item.id for item in obligations.get(coverage.id, [])
                       if item.status == "indeterminate" and item.missing_information_ids)
        return ids

    @staticmethod
    def _ambiguous_assessments(active, exclusions, obligations):
        ids = []
        for coverage in active:
            if coverage.status in {"potentially_covered", "indeterminate"}:
                ids.append(coverage.id)
            ids.extend(item.id for item in exclusions.get(coverage.id, []) if item.status == "indeterminate")
            ids.extend(item.id for item in obligations.get(coverage.id, []) if item.status == "indeterminate")
        return ids

    @staticmethod
    def _result(status, reasons, supporting=None, *, human_review=False, amount=None, currency=None):
        unique_ids = list(dict.fromkeys(supporting or []))
        return ClaimRecommendation(
            id=uuid.uuid4(), status=status, recommended_payable_amount=amount,
            currency=currency if amount is not None else None,
            reasons=list(dict.fromkeys(reasons)), supporting_assessment_ids=unique_ids,
            human_review_required=human_review, disclaimer=DISCLAIMER,
        )
