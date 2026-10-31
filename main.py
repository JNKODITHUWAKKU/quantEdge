"""Run Quant Edge's fixed Sample GMV research from the bundled local inputs."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify an existing run without computing models.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    root = Path(__file__).resolve().parent
    try:
        from src.evidence import reconcile_source_issues, verify_evidence
        from src.runner import current_result, run_research
        from src.source_quality import audit_local_sources

        evidence = verify_evidence(root)
        issues, _ = audit_local_sources(root)
        reconcile_source_issues(issues, evidence["original_source_issues"])
        logging.info(
            "Source audit reproduced %d warnings; original validation evidence verified.",
            int(issues.severity.eq("WARNING").sum()),
        )

        # --verify only checks existing flat CSV digests; default runs or reuses verified results.
        output = current_result(root) if args.verify else run_research(root)
        if output is None:
            logging.error("No completed run exists. Run python main.py first.")
            return 1
        logging.info("%s: %s", "Verified" if args.verify else "Completed", output)
        logging.info("Open the dashboard with: python -m streamlit run app.py")
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, ImportError) as exc:
        logging.error("Research could not complete: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
