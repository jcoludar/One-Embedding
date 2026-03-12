"""Unified benchmark aggregator for scoring PLMs across multiple tasks.

Collects results from retrieval, per-residue probes, biological correlations,
and efficiency metrics, producing normalized scores and summary tables.
"""

import json
import numpy as np
from collections import defaultdict
from pathlib import Path


class BenchmarkAggregator:
    """Aggregates benchmark results across PLMs and tasks.

    Normalization: rank-based within each task (1.0 = best PLM, 0.0 = worst).
    Composite: mean of category means to prevent categories with more tasks from dominating.
    """

    # Task registry: (category, metric_name, higher_is_better)
    TASK_REGISTRY = {
        # Retrieval
        "scop_family_ret1": ("retrieval", "Ret@1", True),
        "scop_sf_ret1": ("retrieval", "SF Ret@1", True),
        "scop_fold_ret1": ("retrieval", "Fold Ret@1", True),
        # Hierarchy
        "scop_separation": ("hierarchy", "Sep. Ratio", True),
        # Per-residue
        "ss3_q3": ("per_residue", "SS3 Q3", True),
        "ss8_q8": ("per_residue", "SS8 Q8", True),
        "disorder_rho": ("per_residue", "Disorder rho", True),
        "tm_f1": ("per_residue", "TM F1", True),
        "signalp_f1": ("per_residue", "SignalP F1", True),
        "ppi_f1": ("per_residue", "PPI F1", True),
        "epitope_f1": ("per_residue", "Epitope F1", True),
        # Biology
        "go_rho": ("biology", "GO rho", True),
        "ec_ret1": ("biology", "EC Ret@1", True),
        "pfam_ret1": ("biology", "Pfam Ret@1", True),
        "taxonomy_rho": ("biology", "Tax. rho", True),
        # Regression
        "solubility_rho": ("regression", "Solub. rho", True),
        "thermostability_rho": ("regression", "Thermo. rho", True),
        # Efficiency
        "embedding_dim": ("efficiency", "Emb. Dim", False),  # lower is better
        "extraction_speed": ("efficiency", "Prot/sec", True),  # higher is better
    }

    def __init__(self):
        # {plm: {task_key: value}}
        self._results: dict[str, dict[str, float]] = defaultdict(dict)

    def add_result(self, plm: str, task: str, value: float):
        """Add a single benchmark result.

        Args:
            plm: PLM name (e.g., "prot_t5_xl", "esm2_650m", "esmc_300m")
            task: Task key from TASK_REGISTRY (e.g., "scop_family_ret1", "ss3_q3")
            value: Metric value
        """
        if task not in self.TASK_REGISTRY:
            print(f"  Warning: unknown task '{task}', adding anyway")
        self._results[plm][task] = value

    def add_results_dict(self, plm: str, results: dict[str, float]):
        """Add multiple results at once."""
        for task, value in results.items():
            self.add_result(plm, task, value)

    def normalized_scores(self) -> dict[str, dict[str, float]]:
        """Rank-based normalization: 1.0 = best PLM, 0.0 = worst.

        For tasks where lower is better, ranking is inverted.
        With 3 PLMs: best=1.0, middle=0.5, worst=0.0
        """
        plms = sorted(self._results.keys())
        if len(plms) < 2:
            # Can't normalize with < 2 PLMs
            return {plm: {task: 1.0 for task in self._results[plm]} for plm in plms}

        all_tasks = set()
        for plm_results in self._results.values():
            all_tasks.update(plm_results.keys())

        normalized = {plm: {} for plm in plms}

        for task in sorted(all_tasks):
            # Get values for PLMs that have this task
            task_plms = [(plm, self._results[plm][task]) for plm in plms if task in self._results[plm]]
            if len(task_plms) < 2:
                for plm, val in task_plms:
                    normalized[plm][task] = 1.0
                continue

            # Determine direction
            higher_is_better = True
            if task in self.TASK_REGISTRY:
                higher_is_better = self.TASK_REGISTRY[task][2]

            # Sort by value
            sorted_plms = sorted(task_plms, key=lambda x: x[1], reverse=higher_is_better)
            n = len(sorted_plms)
            for rank, (plm, val) in enumerate(sorted_plms):
                normalized[plm][task] = 1.0 - rank / (n - 1) if n > 1 else 1.0

        return normalized

    def composite_score(self, weights: dict[str, float] | None = None) -> dict[str, float]:
        """Compute category-balanced composite score per PLM.

        Default: equal weight per category. Mean of category means.

        Args:
            weights: Optional {category: weight}. Default: all categories equal.
        """
        norm = self.normalized_scores()
        plms = sorted(norm.keys())

        # Group tasks by category
        categories = defaultdict(list)
        for task, (cat, _, _) in self.TASK_REGISTRY.items():
            categories[cat].append(task)

        if weights is None:
            weights = {cat: 1.0 for cat in categories}

        composites = {}
        for plm in plms:
            cat_means = {}
            for cat, tasks in categories.items():
                scores = [norm[plm][t] for t in tasks if t in norm[plm]]
                if scores:
                    cat_means[cat] = np.mean(scores)

            if cat_means:
                total_weight = sum(weights.get(cat, 1.0) for cat in cat_means)
                composites[plm] = sum(
                    cat_means[cat] * weights.get(cat, 1.0) for cat in cat_means
                ) / total_weight
            else:
                composites[plm] = 0.0

        return composites

    def summary_table(self) -> str:
        """Generate a Markdown summary table."""
        plms = sorted(self._results.keys())
        if not plms:
            return "No results added yet."

        # Group tasks by category
        categories = defaultdict(list)
        all_tasks = set()
        for plm_results in self._results.values():
            all_tasks.update(plm_results.keys())

        for task in sorted(all_tasks):
            if task in self.TASK_REGISTRY:
                cat = self.TASK_REGISTRY[task][0]
            else:
                cat = "other"
            categories[cat].append(task)

        lines = []
        header = "| Task | " + " | ".join(plms) + " |"
        sep = "|------|" + "|".join(["------"] * len(plms)) + "|"
        lines.append(header)
        lines.append(sep)

        for cat in ["retrieval", "hierarchy", "per_residue", "biology", "regression", "efficiency", "other"]:
            if cat not in categories:
                continue
            lines.append(f"| **{cat.replace('_', ' ').title()}** | " + " | ".join([""] * len(plms)) + " |")
            for task in sorted(categories[cat]):
                display_name = self.TASK_REGISTRY.get(task, ("", task, True))[1]
                vals = []
                for plm in plms:
                    v = self._results[plm].get(task)
                    if v is not None:
                        vals.append(f"{v:.3f}")
                    else:
                        vals.append("-")
                lines.append(f"| {display_name} | " + " | ".join(vals) + " |")

        # Composite scores
        composites = self.composite_score()
        lines.append(f"| **Composite** | " + " | ".join(
            f"**{composites.get(plm, 0):.3f}**" for plm in plms
        ) + " |")

        return "\n".join(lines)

    def plot_radar(self, output_path: str, title: str = "PLM Benchmark Radar"):
        """Generate a radar/spider chart comparing PLMs across categories.

        Uses matplotlib. One axis per category (mean of normalized scores in that category).
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        norm = self.normalized_scores()
        plms = sorted(norm.keys())

        # Get category means for each PLM
        categories_order = ["retrieval", "hierarchy", "per_residue", "biology", "regression", "efficiency"]
        cat_labels = ["Retrieval", "Hierarchy", "Per-residue", "Biology", "Regression", "Efficiency"]

        # Filter to categories that have data
        active_cats = []
        active_labels = []
        for cat, label in zip(categories_order, cat_labels):
            tasks = [t for t, (c, _, _) in self.TASK_REGISTRY.items() if c == cat]
            has_data = any(
                any(t in norm[plm] for t in tasks)
                for plm in plms
            )
            if has_data:
                active_cats.append(cat)
                active_labels.append(label)

        if not active_cats:
            print("  No data for radar plot")
            return

        N = len(active_cats)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # close the polygon

        fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw=dict(polar=True))

        colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0"]

        for i, plm in enumerate(plms):
            values = []
            for cat in active_cats:
                tasks = [t for t, (c, _, _) in self.TASK_REGISTRY.items() if c == cat]
                scores = [norm[plm][t] for t in tasks if t in norm[plm]]
                values.append(np.mean(scores) if scores else 0.0)
            values += values[:1]  # close

            color = colors[i % len(colors)]
            ax.plot(angles, values, 'o-', linewidth=2, label=plm, color=color)
            ax.fill(angles, values, alpha=0.1, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(active_labels, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=14, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
        ax.grid(True, alpha=0.3)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Radar plot saved to {output_path}")

    def to_json(self) -> dict:
        """Export full results as JSON-serializable dict."""
        norm = self.normalized_scores()
        composites = self.composite_score()
        return {
            "raw_results": dict(self._results),
            "normalized_scores": norm,
            "composite_scores": composites,
            "summary_table": self.summary_table(),
        }

    def save(self, path: str):
        """Save results to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_json(), f, indent=2, default=str)
        print(f"  Aggregator results saved to {path}")
