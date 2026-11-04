"""Create an explicit, hash-bound reference contract for subsequent candidates."""
import argparse
import json
from pathlib import Path

from compile_aot import file_hash, resolve_tolerances, validate_reference_contract


def build(baseline, reference=None, authorization=None, rtol=None, atol=None):
    if reference is None:
        if authorization:
            raise ValueError("Authorization requires an explicitly captured baseline reference")
        rtol, atol = resolve_tolerances(rtol, atol, baseline["request"])
        contract = {"kind": "validated_reference", "passed": True,
                    "reference": baseline["reference"]}
    else:
        if not authorization or not authorization.strip():
            raise ValueError("Baseline equivalence needs this task's explicit user authorization")
        rtol, atol = resolve_tolerances(rtol, atol)
        reference = Path(reference).resolve(strict=True)
        contract = {"kind": "baseline_equivalence", "reference": {
            "path": str(reference), "sha256": file_hash(reference)},
            "diagnostic_timing_authorization": authorization,
            "scope": "Equivalence to the compiled baseline; not independent model accuracy"}
    contract.update(package_sha256=baseline["package"]["sha256"],
                    inputs_sha256=baseline["saved_inputs"]["sha256"],
                    torch_backends=baseline["request"]["torch_backends"], rtol=rtol, atol=atol)
    validate_reference_contract(baseline, contract)
    for artifact in (baseline["package"], baseline["saved_inputs"], contract["reference"]):
        if file_hash(artifact["path"]) != artifact["sha256"]:
            raise ValueError("Immutable baseline/reference artifact changed")
    return contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference", type=Path, help="Explicit saved outputs of compiled baseline on its exact saved inputs")
    parser.add_argument("--authorization", help="Quote or link this task's user instruction accepting baseline equivalence")
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--atol", type=float)
    args = parser.parse_args()
    baseline = json.loads((args.baseline_run / "manifest.json").read_text())
    contract = build(baseline, args.reference, args.authorization, args.rtol, args.atol)
    with args.output.open("x") as stream:
        json.dump(contract, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
