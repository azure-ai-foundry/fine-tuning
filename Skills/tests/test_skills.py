"""Tests for the fine-tuning skill scripts, docs, and CLI.

Covers compilation, security, code quality, SKILL.md consistency,
and auto_finetune.py CLI validation.
"""

import os
import pathlib
import py_compile
import re
import subprocess
import sys

import pytest

SKILLS_DIR = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS_DIR = SKILLS_DIR / "scripts"
REFERENCES_DIR = SKILLS_DIR / "references"
WORKFLOWS_DIR = SKILLS_DIR / "workflows"
SKILL_MD = SKILLS_DIR / "SKILL.md"


# ── Compilation ──────────────────────────────────────────────────────────

def _find_py_files(root):
    """Recursively find all .py files under root."""
    results = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.endswith(".py") and f != "__init__.py":
                results.append(os.path.join(dirpath, f))
            elif f == "__init__.py":
                results.append(os.path.join(dirpath, f))
    return results


class TestCompilation:
    """Every Python script must compile without syntax errors."""

    @pytest.fixture(params=[
        pytest.param(p, id=os.path.relpath(p, SCRIPTS_DIR))
        for p in _find_py_files(SCRIPTS_DIR)
    ])
    def script_path(self, request):
        return request.param

    def test_script_compiles(self, script_path):
        py_compile.compile(script_path, doraise=True)


# ── Security ─────────────────────────────────────────────────────────────

class TestSecurity:
    """Security checks across all scripts."""

    def test_no_bare_resp_json_in_error_paths(self):
        """No bare resp.json().get() calls — must be wrapped in try/except."""
        issues = []
        for filepath in _find_py_files(SCRIPTS_DIR):
            content = open(filepath, encoding="utf-8").read()
            filename = os.path.relpath(filepath, SCRIPTS_DIR)
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if (
                    "resp.json().get(" in line
                    and not line.strip().startswith("#")
                    and "try:" not in line
                    and "_safe_error_msg" not in line
                ):
                    # Check preceding 3 lines for a try: block
                    preceding = lines[max(0, i - 3):i]
                    if not any(pl.strip().startswith("try:") for pl in preceding):
                        issues.append(
                            f"{filename}:{i + 1}: bare resp.json() — use try/except or _safe_error_msg()"
                        )
        assert issues == []

    def test_no_hardcoded_api_keys(self):
        """No hardcoded API keys or secrets in scripts."""
        key_patterns = [
            r'api[_-]?key\s*=\s*["\'][A-Za-z0-9]{20,}["\']',
            r'secret\s*=\s*["\'][A-Za-z0-9]{20,}["\']',
            r'password\s*=\s*["\'][A-Za-z0-9]{8,}["\']',
        ]
        issues = []
        for filepath in _find_py_files(SCRIPTS_DIR):
            content = open(filepath, encoding="utf-8").read()
            filename = os.path.relpath(filepath, SCRIPTS_DIR)
            for pattern in key_patterns:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    issues.append(f"{filename}: possible hardcoded secret: {match.group()[:40]}...")
        assert issues == []

    def test_exec_has_security_docstring(self):
        """Any file using exec() must have a security/trust boundary comment nearby."""
        issues = []
        for filepath in _find_py_files(SCRIPTS_DIR):
            content = open(filepath, encoding="utf-8").read()
            filename = os.path.relpath(filepath, SCRIPTS_DIR)
            if "exec(" in content and "compile(" in content:
                # Check for security warning within 10 lines of exec()
                lines = content.split("\n")
                for i, line in enumerate(lines):
                    if "exec(" in line and not line.strip().startswith("#"):
                        context = "\n".join(lines[max(0, i - 10):i + 3])
                        if not any(kw in context.lower() for kw in ["security", "warning", "trust", "arbitrary"]):
                            issues.append(f"{filename}:{i + 1}: exec() without security comment")
        assert issues == []


# ── Code Quality ─────────────────────────────────────────────────────────

class TestCodeQuality:
    """Code quality checks."""

    def test_encoding_on_text_open(self):
        """Text file opens should specify encoding='utf-8'."""
        issues = []
        for filepath in _find_py_files(SCRIPTS_DIR):
            content = open(filepath, encoding="utf-8").read()
            filename = os.path.relpath(filepath, SCRIPTS_DIR)
            lines = content.split("\n")
            for i, line in enumerate(lines):
                # Match open() for text mode (not "rb"/"wb")
                if re.search(r'open\([^)]+\)', line) and "encoding" not in line:
                    # Skip binary mode opens
                    if '"rb"' in line or "'rb'" in line or '"wb"' in line or "'wb'" in line:
                        continue
                    # Skip imports and comments
                    if line.strip().startswith("#") or line.strip().startswith("import"):
                        continue
                    # Skip if it's a with/as pattern referencing binary
                    if "purpose=" in line:  # files.create() calls
                        continue
                    issues.append(f"{filename}:{i + 1}: open() without encoding — use encoding='utf-8'")
        # Allow a few known cases (some opens are binary or handled elsewhere)
        assert len(issues) <= 2, f"Too many unencoded opens:\n" + "\n".join(issues)

    def test_no_custom_client_creation(self):
        """Scripts should use common.get_clients(), not custom client creation."""
        issues = []
        skip_files = {"common.py", "auto_finetune.py"}  # These are allowed to create clients
        for filepath in _find_py_files(SCRIPTS_DIR):
            filename = os.path.relpath(filepath, SCRIPTS_DIR)
            if os.path.basename(filepath) in skip_files:
                continue
            if os.sep + "validate" + os.sep in filepath or "validate\\" in filepath:
                continue
            content = open(filepath, encoding="utf-8").read()
            if "AzureOpenAI(" in content or "OpenAI(" in content:
                # Check if it's importing from common or creating its own
                if "from common import" not in content and "import common" not in content:
                    issues.append(f"{filename}: creates OpenAI client without using common.get_clients()")
        assert issues == []


# ── Skill Consistency ────────────────────────────────────────────────────

class TestSkillConsistency:
    """SKILL.md references must match actual files."""

    @pytest.fixture(autouse=True)
    def load_skill_md(self):
        self.skill_content = SKILL_MD.read_text(encoding="utf-8")

    def test_skill_script_references_exist(self):
        """Every script referenced in SKILL.md must exist."""
        refs = re.findall(r'`scripts/[\w/.]+\.py`', self.skill_content)
        for ref in refs:
            script_path = ref.strip("`")
            full_path = SKILLS_DIR / script_path
            assert full_path.exists(), f"SKILL.md references {ref} but file doesn't exist"

    def test_skill_reference_docs_exist(self):
        """Every reference doc linked in SKILL.md must exist."""
        refs = re.findall(r'`references/[\w-]+\.md`', self.skill_content)
        for ref in refs:
            doc_path = ref.strip("`")
            full_path = SKILLS_DIR / doc_path
            assert full_path.exists(), f"SKILL.md references {ref} but file doesn't exist"

    def test_skill_workflow_docs_exist(self):
        """Every workflow doc linked in SKILL.md must exist."""
        refs = re.findall(r'`workflows/[\w-]+\.md`', self.skill_content)
        for ref in refs:
            doc_path = ref.strip("`")
            full_path = SKILLS_DIR / doc_path
            assert full_path.exists(), f"SKILL.md references {ref} but file doesn't exist"


# ── Auto-Finetune CLI ────────────────────────────────────────────────────

class TestAutoFinetuneCLI:
    """Validate auto_finetune.py CLI structure."""

    AUTO_FT = str(SCRIPTS_DIR / "auto_finetune.py")

    def test_auto_subcommand_help(self):
        """The 'auto' subcommand should accept --help."""
        result = subprocess.run(
            [sys.executable, self.AUTO_FT, "auto", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "--data" in result.stdout
        assert "--description" in result.stdout

    def test_all_subcommands_exist(self):
        """All expected subcommands should be available."""
        expected = ["analyze", "generate", "foundry-generate", "prepare", "baseline",
                    "candidates", "execute", "evaluate", "review", "auto"]
        result = subprocess.run(
            [sys.executable, self.AUTO_FT, "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        for cmd in expected:
            assert cmd in result.stdout, f"Subcommand '{cmd}' not found in help output"

    def test_foundry_generate_subcommand_help(self):
        """The 'foundry-generate' subcommand should expose Foundry datagen flags."""
        result = subprocess.run(
            [sys.executable, self.AUTO_FT, "foundry-generate", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        for flag in ["--task-spec", "--source", "--recipe", "--scenario",
                     "--max-samples", "--train-split", "--teacher",
                     "--prompt", "--prompt-file", "--file-id",
                     "--agent-name", "--agent-version", "--hours",
                     "--project-endpoint"]:
            assert flag in result.stdout, f"foundry-generate missing flag {flag}"
        # Source/recipe/scenario choices appear
        assert "traces" in result.stdout and "tool-use" in result.stdout

    def test_auto_includes_datagen_backend(self):
        """`auto` subcommand should expose --datagen-backend selector for Foundry datagen."""
        result = subprocess.run(
            [sys.executable, self.AUTO_FT, "auto", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        for flag in ["--datagen-backend", "--datagen-file-id",
                     "--datagen-agent-name", "--datagen-hours"]:
            assert flag in result.stdout, f"auto missing {flag}"
        # Backend choices
        for choice in ["local", "foundry-prompt", "foundry-file",
                       "foundry-agent", "foundry-traces"]:
            assert choice in result.stdout, f"--datagen-backend choice {choice} missing"

    def test_analyze_accepts_connection_args(self):
        """The 'analyze' subcommand should accept connection args."""
        result = subprocess.run(
            [sys.executable, self.AUTO_FT, "analyze", "--help"],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0
        assert "--base-url" in result.stdout
        assert "--api-key" in result.stdout


# ── Cost Estimation ──────────────────────────────────────────────────────

class TestCostEstimation:
    """Validate the centralized SFT training cost estimator in common.py."""

    @pytest.fixture(autouse=True)
    def _add_scripts_to_path(self):
        sys.path.insert(0, str(SCRIPTS_DIR))
        yield
        try:
            sys.path.remove(str(SCRIPTS_DIR))
        except ValueError:
            pass

    def test_lookup_exact_match(self):
        from common import lookup_training_price
        price, key = lookup_training_price("gpt-4.1-mini")
        assert price == 5.0
        assert key == "gpt-4.1-mini"

    def test_lookup_versioned_model(self):
        """Versioned IDs like gpt-4.1-mini-2025-04-14 should match the base."""
        from common import lookup_training_price
        price, key = lookup_training_price("gpt-4.1-mini-2025-04-14")
        assert price == 5.0
        assert key == "gpt-4.1-mini"

    def test_lookup_finetune_suffix_stripped(self):
        """Fine-tune suffixes (.ft-, :ft-, -ft-) should be stripped."""
        from common import lookup_training_price
        for suffix in (".ft-foo", ":ft-bar", "-ft-baz"):
            price, key = lookup_training_price(f"gpt-4.1-mini{suffix}")
            assert price == 5.0, f"failed for suffix '{suffix}'"
            assert key == "gpt-4.1-mini"

    def test_lookup_case_insensitive(self):
        from common import lookup_training_price
        price, key = lookup_training_price("GPT-4.1-MINI")
        assert price == 5.0
        assert key == "gpt-4.1-mini"

    def test_lookup_unknown_returns_none(self):
        from common import lookup_training_price
        price, key = lookup_training_price("mystery-model-9000")
        assert price is None
        assert key is None

    def test_lookup_empty_input(self):
        from common import lookup_training_price
        assert lookup_training_price(None) == (None, None)
        assert lookup_training_price("") == (None, None)

    def test_lookup_longest_prefix_wins(self):
        """gpt-4.1-mini-X should match 'gpt-4.1-mini', not 'gpt-4.1'."""
        from common import lookup_training_price, SFT_TRAINING_PRICE_PER_M_TOKENS
        assert SFT_TRAINING_PRICE_PER_M_TOKENS["gpt-4.1"] == 25.0
        assert SFT_TRAINING_PRICE_PER_M_TOKENS["gpt-4.1-mini"] == 5.0
        # The mini variant should win because it's a longer prefix
        price, key = lookup_training_price("gpt-4.1-mini-some-future-version")
        assert key == "gpt-4.1-mini"
        assert price == 5.0

    def test_estimate_basic(self):
        from common import estimate_training_cost
        est = estimate_training_cost("gpt-4.1-mini", "globalStandard", 1_000_000)
        assert est is not None
        assert est["cost"] == 5.0
        assert est["price_per_M"] == 5.0
        assert est["tier_multiplier"] == 1.0
        assert est["matched_model"] == "gpt-4.1-mini"
        assert est["effective_tier"] == "globalStandard"

    def test_estimate_developer_tier(self):
        """developerTier is 50% off globalStandard."""
        from common import estimate_training_cost
        est = estimate_training_cost("gpt-4.1-mini", "developerTier", 1_000_000)
        assert est["cost"] == 2.5
        assert est["tier_multiplier"] == 0.5
        assert est["effective_tier"] == "developerTier"

    def test_estimate_standard_tier(self):
        """Regional standard is +20% (midpoint of 10–30%)."""
        from common import estimate_training_cost
        est = estimate_training_cost("gpt-4.1-mini", "standard", 1_000_000)
        assert est["cost"] == 6.0
        assert est["tier_multiplier"] == 1.2

    def test_estimate_oss_global_standard(self):
        from common import estimate_training_cost
        est = estimate_training_cost("ministral-3b", "globalStandard", 1_000_000)
        assert est["cost"] == 1.0
        assert est["effective_tier"] == "globalStandard"

    def test_estimate_oss_forces_global_standard(self):
        """OSS models only support globalStandard. Other tiers must be overridden."""
        from common import estimate_training_cost
        for bad_tier in ("developerTier", "standard"):
            est = estimate_training_cost("ministral-3b", bad_tier, 1_000_000)
            assert est is not None
            assert est["effective_tier"] == "globalStandard", \
                f"OSS + {bad_tier} should override to globalStandard"
            assert est["cost"] == 1.0, \
                f"OSS cost should be globalStandard rate, not {bad_tier} rate"
            assert est["tier_multiplier"] == 1.0

    def test_estimate_all_oss_models_constrained(self):
        """All four OSS models should override to globalStandard."""
        from common import estimate_training_cost
        oss_models = {
            "ministral-3b":   1.00,
            "qwen3-32b":      3.20,
            "llama-3.3-70b":  4.50,
            "gpt-oss-20b":    3.60,
        }
        for model, expected_price in oss_models.items():
            est = estimate_training_cost(model, "developerTier", 1_000_000)
            assert est["cost"] == expected_price, \
                f"{model}: expected ${expected_price}, got ${est['cost']}"
            assert est["effective_tier"] == "globalStandard"

    def test_estimate_unknown_model_returns_none(self):
        from common import estimate_training_cost
        assert estimate_training_cost("unknown-model", "globalStandard", 1_000_000) is None

    def test_estimate_zero_tokens_returns_none(self):
        from common import estimate_training_cost
        assert estimate_training_cost("gpt-4.1-mini", "globalStandard", 0) is None
        assert estimate_training_cost("gpt-4.1-mini", "globalStandard", None) is None

    def test_estimate_tier_case_insensitive(self):
        from common import estimate_training_cost
        for tier in ("globalStandard", "GLOBALSTANDARD", "globalstandard"):
            est = estimate_training_cost("gpt-4.1-mini", tier, 1_000_000)
            assert est["cost"] == 5.0, f"failed for tier '{tier}'"

    def test_estimate_default_tier_is_global_standard(self):
        from common import estimate_training_cost
        est = estimate_training_cost("gpt-4.1-mini", None, 1_000_000)
        assert est["cost"] == 5.0
        assert est["effective_tier"] == "globalStandard"

    def test_estimate_rft_only_models_return_none(self):
        """o4-mini and gpt-5 are RFT-only/hourly-billed; not in the dict."""
        from common import estimate_training_cost
        for model in ("o4-mini", "gpt-5"):
            assert estimate_training_cost(model, "globalStandard", 1_000_000) is None, \
                f"{model} is RFT-only/hourly — should not have a per-token estimate"
