"""Thin REST client for the FinRL Pro control plane API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping

import requests


class FinRLProAPIError(RuntimeError):
    """Raised when the control plane API returns a non-success status."""


@dataclass(slots=True)
class FinRLProAPIClient:
    """HTTP client wrapping FinRL Pro control plane endpoints."""

    base_url: str
    session: requests.Session = field(default_factory=requests.Session)

    def _request(
        self,
        method: str,
        path: str,
        *,
        json_payload: Mapping[str, Any] | None = None,
    ) -> MutableMapping[str, Any]:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        response = self.session.request(method=method, url=url, json=json_payload)
        if response.status_code >= 400:
            raise FinRLProAPIError(
                f"{method} {url} failed with {response.status_code}: {response.text}"
            )
        if not response.content:
            return {}
        return response.json()

    def create_experiment(self, payload: Mapping[str, Any]) -> MutableMapping[str, Any]:
        """Register a new experiment fingerprint and launch training."""
        return self._request("POST", "/experiments", json_payload=payload)

    def trigger_reproduction(self, fingerprint_id: str) -> MutableMapping[str, Any]:
        """Trigger reproducibility validation for a fingerprint."""
        return self._request(
            "POST", f"/experiments/{fingerprint_id}/reproduce", json_payload={}
        )

    def create_evaluation(self, payload: Mapping[str, Any]) -> MutableMapping[str, Any]:
        """Launch an evaluation campaign using the benchmark catalog."""
        return self._request("POST", "/evaluations", json_payload=payload)

    def upsert_risk_profile(self, payload: Mapping[str, Any]) -> MutableMapping[str, Any]:
        """Create or update a risk control profile."""
        return self._request("POST", "/risk-profiles", json_payload=payload)

    def get_risk_profile(self, profile_id: str) -> MutableMapping[str, Any]:
        """Fetch a risk control profile by identifier."""
        return self._request("GET", f"/risk-profiles/{profile_id}")

    def create_report(self, payload: Mapping[str, Any]) -> MutableMapping[str, Any]:
        """Publish a compliance-ready performance report."""
        return self._request("POST", "/reports", json_payload=payload)

    def get_report(self, report_id: str) -> MutableMapping[str, Any]:
        """Retrieve report metadata and approval status."""
        return self._request("GET", f"/reports/{report_id}")
