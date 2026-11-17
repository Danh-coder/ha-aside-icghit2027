"""Paper safety protocol and explicitly named compatibility with ASIDE's source.

The original source contains defective regex selection and output normalization.
Keep those observable for comparisons rather than silently calling a fix identical.
"""

import ast
from functools import lru_cache
import json
import warnings
from .common import ORIGINAL

SAFETY_REVISION = "aside_safety_v4"
DIRECT_SUITES = frozenset({"tensortrust", "gandalf", "purple", "rules"})
INDIRECT_SUITES = frozenset({"struq_id", "struq_ood", "bipia_text", "bipia_code"})
SAFETY_SUITES = DIRECT_SUITES | INDIRECT_SUITES
BIPIA_FILTERS = {"aside_source_regex", "programmatic_literal"}
BIPIA_TEXT_SCENARIOS = ["email", "table", "abstract", "qa"]
DIRECT_COUNTS = {"tensortrust": 165, "gandalf": 140, "purple": 285, "rules": 285}


def expected_safety_count(suite, bipia_filter="aside_source_regex"):
    if suite in DIRECT_COUNTS:
        return DIRECT_COUNTS[suite]
    if suite in {"struq_id", "struq_ood"}:
        return 208 * 4
    if suite == "bipia_text":
        return 12276
    if suite == "bipia_code":
        return 6000 if bipia_filter == "aside_source_regex" else 7500
    return None


@lru_cache(maxsize=1)
def original_template():
    template = json.loads((ORIGINAL / "data/prompt_templates.json").read_text(encoding="utf-8"))[0]
    if any(template[key].count("{}") != 1 for key in ["system", "user"]):
        raise ValueError("The original prompt template must contain one content field per role")
    return template


@lru_cache(maxsize=2)
def _bipia_filter_expression(policy):
    if policy not in BIPIA_FILTERS:
        raise ValueError(f"Unknown BIPIA attack selection: {policy}")
    path = ORIGINAL / "evaluations/safety_evals/test_safety.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "evaluate_scenario")
    selection = next(
        n.value.slice for n in function.body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "df" for t in n.targets)
        and isinstance(n.value, ast.Subscript)
        and isinstance(n.value.value, ast.Name) and n.value.value.id == "pia_samples"
    )
    if policy == "programmatic_literal":
        class LiteralContains(ast.NodeTransformer):
            def visit_Call(self, node):
                self.generic_visit(node)
                if isinstance(node.func, ast.Attribute) and node.func.attr == "contains":
                    node.keywords.append(ast.keyword(arg="regex", value=ast.Constant(False)))
                return node
        selection = LiteralContains().visit(selection)
    return compile(ast.fix_missing_locations(ast.Expression(selection)), str(path), "eval")


def bipia_attack_mask(frame, policy="aside_source_regex"):
    """Execute only the original selection expression, not its model-loading script."""
    with warnings.catch_warnings():
        # The capturing groups are intentional here because compatibility must
        # preserve the original regex semantics, including its excluded attacks.
        warnings.filterwarnings("ignore", message="This pattern is interpreted as a regular expression.*", category=UserWarning)
        return eval(_bipia_filter_expression(policy), {"__builtins__": {}}, {"pia_samples": frame})


def generation_settings(suite, config):
    sampling = suite in DIRECT_SUITES
    settings = {
        "do_sample": sampling,
        "temperature": 0.7 if sampling else None,
        "top_k": config.get("direct_top_k", 50) if sampling else 0,
        "top_p": 1.0,
        "max_new_tokens": config["max_new_tokens"],
        "num_beams": 1,
    }
    return settings
