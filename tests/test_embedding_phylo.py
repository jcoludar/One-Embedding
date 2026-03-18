"""Tests for Experiment 35: Bayesian phylogenetics from protein embeddings."""

import sys
import importlib
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

_exp35 = importlib.import_module("35_embedding_phylogenetics")
TreeNode = _exp35.TreeNode
Tree = _exp35.Tree
parse_newick = _exp35.parse_newick
write_newick = _exp35.write_newick
NJBuilder = _exp35.NJBuilder
random_tree = _exp35.random_tree
BMLikelihood = _exp35.BMLikelihood
StochasticNNI = _exp35.StochasticNNI
SubtreePruneRegraft = _exp35.SubtreePruneRegraft
BranchLengthMultiplier = _exp35.BranchLengthMultiplier
TreeLengthMultiplier = _exp35.TreeLengthMultiplier
SigmaMultiplier = _exp35.SigmaMultiplier
ProposalMixer = _exp35.ProposalMixer
simulate_bm = _exp35.simulate_bm
MCMCChain = _exp35.MCMCChain
MC3Runner = _exp35.MC3Runner
MultiRunOrchestrator = _exp35.MultiRunOrchestrator
Diagnostics = _exp35.Diagnostics
ConsensusBuilder = _exp35.ConsensusBuilder
robinson_foulds = _exp35.robinson_foulds
estimate_sigma2 = _exp35.estimate_sigma2
normalize_leaf_names = _exp35.normalize_leaf_names
BL_MIN = _exp35.BL_MIN
BL_MAX = _exp35.BL_MAX
clamp_branch_length = _exp35.clamp_branch_length
NodeSlider = _exp35.NodeSlider
load_msa = _exp35.load_msa
extract_aligned_embeddings = _exp35.extract_aligned_embeddings
evaluate_monophyly = _exp35.evaluate_monophyly
clade_purity_score = _exp35.clade_purity_score
family_separation_score = _exp35.family_separation_score


# ---------------------------------------------------------------------------
# TestTreeNode
# ---------------------------------------------------------------------------

class TestTreeNode:
    """Tests for the TreeNode dataclass."""

    def test_leaf_creation(self):
        """A leaf node should have is_leaf=True, correct name and branch_length."""
        leaf = TreeNode(id=0, name="ProtA", branch_length=0.05)
        assert leaf.is_leaf()
        assert leaf.name == "ProtA"
        assert leaf.branch_length == 0.05
        assert leaf.children == []

    def test_internal_node(self):
        """An internal node with children should not be a leaf."""
        child_a = TreeNode(id=1, name="A")
        child_b = TreeNode(id=2, name="B")
        internal = TreeNode(id=0, children=[child_a, child_b])
        assert not internal.is_leaf()
        assert len(internal.children) == 2


# ---------------------------------------------------------------------------
# TestTree
# ---------------------------------------------------------------------------

class TestTree:
    """Tests for the Tree class."""

    def _make_3taxon_tree(self):
        """Build a simple 3-taxon tree: ((A,B),C)."""
        root = TreeNode(id=0)
        internal = TreeNode(id=1, branch_length=0.1, parent=root)
        leaf_a = TreeNode(id=2, name="A", branch_length=0.2, parent=internal)
        leaf_b = TreeNode(id=3, name="B", branch_length=0.3, parent=internal)
        leaf_c = TreeNode(id=4, name="C", branch_length=0.4, parent=root)
        internal.children = [leaf_a, leaf_b]
        root.children = [internal, leaf_c]
        return Tree(root)

    def test_structure(self):
        """3-taxon tree should have 3 leaves and 2 internal nodes."""
        tree = self._make_3taxon_tree()
        assert tree.n_leaves == 3
        assert set(tree.leaf_names()) == {"A", "B", "C"}
        assert tree.n_internal == 2  # root + one internal

    def test_copy_independence(self):
        """Modifying a copy should not affect the original."""
        tree = self._make_3taxon_tree()
        tree_copy = tree.copy()

        # Modify copy's root branch length.
        tree_copy.root.children[0].branch_length = 999.0

        # Original should be unchanged.
        assert tree.root.children[0].branch_length == 0.1


# ---------------------------------------------------------------------------
# TestNewickIO
# ---------------------------------------------------------------------------

class TestNewickIO:
    """Tests for Newick parsing and writing."""

    def test_simple_roundtrip(self):
        """Parse a simple Newick, write it back, parse again — structure preserved."""
        nwk = "((A:0.1,B:0.2):0.3,C:0.4);"
        tree = parse_newick(nwk)
        assert tree.n_leaves == 3
        assert set(tree.leaf_names()) == {"A", "B", "C"}

        nwk2 = write_newick(tree)
        tree2 = parse_newick(nwk2)
        assert tree2.n_leaves == 3
        assert set(tree2.leaf_names()) == {"A", "B", "C"}

    def test_branch_lengths_preserved(self):
        """Branch lengths should survive parse-write-parse roundtrip."""
        nwk = "((A:0.1234567890,B:0.9876543210):0.5,C:0.3);"
        tree = parse_newick(nwk)

        # Find leaf A.
        leaf_a = [n for n in tree.leaves if n.name == "A"][0]
        assert abs(leaf_a.branch_length - 0.1234567890) < 1e-9

        # Roundtrip.
        nwk2 = write_newick(tree)
        tree2 = parse_newick(nwk2)
        leaf_a2 = [n for n in tree2.leaves if n.name == "A"][0]
        assert abs(leaf_a2.branch_length - 0.1234567890) < 1e-9

    def test_pipe_chars_quoted(self):
        """Names with pipe chars should be handled (quoted in output)."""
        nwk = "('A0A1P8|org1':0.1,'P58917|org2':0.2);"
        tree = parse_newick(nwk)
        assert tree.n_leaves == 2
        names = set(tree.leaf_names())
        assert "A0A1P8|org1" in names
        assert "P58917|org2" in names

        # Write and re-parse.
        nwk2 = write_newick(tree)
        assert "|" in nwk2  # Pipes should be in the output.
        tree2 = parse_newick(nwk2)
        assert set(tree2.leaf_names()) == names

    def test_trifurcating_resolved(self):
        """A trifurcating tree should be resolved to binary by resolve_polytomies."""
        nwk = "(A:0.1,B:0.2,C:0.3);"
        tree = parse_newick(nwk)
        assert tree.n_leaves == 3

        # Root has 3 children (trifurcation).
        assert len(tree.root.children) == 3

        tree.resolve_polytomies()

        # After resolution, root should have exactly 2 children.
        assert len(tree.root.children) == 2

        # Still 3 leaves.
        assert tree.n_leaves == 3

        # All internal nodes should be binary.
        for node in tree.internals:
            assert len(node.children) == 2, (
                f"Node {node.id} has {len(node.children)} children"
            )

    def test_real_iqtree(self):
        """Parse a real IQ-TREE treefile with 40 leaves."""
        treefile = Path(
            "/Users/jcoludar/CascadeProjects/SpeciesEmbedding/results/"
            "iqtree_conotoxin.treefile"
        )
        if not treefile.exists():
            pytest.skip("IQ-TREE treefile not found")

        nwk = treefile.read_text().strip()
        tree = parse_newick(nwk)
        tree.resolve_polytomies()

        assert tree.n_leaves == 40

        # All internal nodes should be binary after resolution.
        for node in tree.internals:
            assert len(node.children) == 2, (
                f"Node {node.id} has {len(node.children)} children"
            )


# ---------------------------------------------------------------------------
# TestNJBuilder
# ---------------------------------------------------------------------------

class TestNJBuilder:
    """Tests for the Neighbor-Joining tree builder."""

    def test_4taxon_textbook(self):
        """Classic 4-taxon NJ example: A and B should be siblings."""
        D = np.array([
            [0, 5, 9, 9],
            [5, 0, 10, 10],
            [9, 10, 0, 8],
            [9, 10, 8, 0],
        ], dtype=float)
        names = ["A", "B", "C", "D"]
        tree = NJBuilder.build(D, names)

        assert tree.n_leaves == 4
        assert set(tree.leaf_names()) == {"A", "B", "C", "D"}

        # A and B should share a parent (be siblings).
        leaf_a = tree._name_to_leaf["A"]
        leaf_b = tree._name_to_leaf["B"]
        assert leaf_a.parent is leaf_b.parent, "A and B should be siblings"

    def test_3taxon_positive_bl(self):
        """A 3-taxon NJ tree should have positive total branch length."""
        D = np.array([
            [0, 3, 5],
            [3, 0, 4],
            [5, 4, 0],
        ], dtype=float)
        names = ["X", "Y", "Z"]
        tree = NJBuilder.build(D, names)

        assert tree.n_leaves == 3
        assert tree.total_branch_length() > 0

    def test_from_embeddings(self):
        """NJ from random 512d embeddings should produce correct leaf set."""
        rng = np.random.default_rng(123)
        embs = {
            "P1": rng.standard_normal(512),
            "P2": rng.standard_normal(512),
            "P3": rng.standard_normal(512),
            "P4": rng.standard_normal(512),
        }
        tree = NJBuilder.from_embeddings(embs)
        assert tree.n_leaves == 4
        assert set(tree.leaf_names()) == {"P1", "P2", "P3", "P4"}


# ---------------------------------------------------------------------------
# TestRandomTree
# ---------------------------------------------------------------------------

class TestRandomTree:
    """Tests for random tree generation."""

    def test_correct_n_leaves(self):
        """Random tree should have the correct number of leaves."""
        names = ["A", "B", "C", "D", "E", "F"]
        tree = random_tree(names)
        assert tree.n_leaves == len(names)
        assert set(tree.leaf_names()) == set(names)

    def test_all_binary(self):
        """All internal nodes should have exactly 2 children."""
        names = ["A", "B", "C", "D", "E"]
        tree = random_tree(names)
        for node in tree.internals:
            assert len(node.children) == 2, (
                f"Node {node.id} has {len(node.children)} children"
            )

    def test_positive_branch_lengths(self):
        """All branch lengths should be > 0."""
        names = ["A", "B", "C", "D", "E", "F", "G"]
        tree = random_tree(names)
        for node in tree.nodes:
            if not node.is_root():
                assert node.branch_length > 0, (
                    f"Node {node.id} ({node.name}) has bl={node.branch_length}"
                )


# ---------------------------------------------------------------------------
# TestBMLikelihood
# ---------------------------------------------------------------------------

class TestBMLikelihood:
    """Tests for Brownian Motion likelihood (Felsenstein pruning)."""

    def test_three_taxon_analytical(self):
        """3-taxon tree, 1 dimension — compare to hand-calculated value.

        Tree: ((A:1,B:1):0,C:2)  (root has zero-length internal branch)
        Data: A=0, B=2, C=1, sigma2=1.0

        Internal node (joining A and B):
          s_L = 0 + 1 = 1 (v_A + t_A)
          s_R = 0 + 1 = 1 (v_B + t_B)
          logL_node1 = -0.5 * 1 * log(2*pi*1.0*(1+1)) - 0.5 * (0-2)^2 / (1.0*(1+1))
                     = -0.5 * log(4*pi) - 0.5 * 4/2
                     = -0.5 * log(4*pi) - 1.0

        Root node (joining internal with C):
          v_internal = 1*1/(1+1) = 0.5
          mu_internal = (0*1 + 2*1)/(1+1) = 1.0
          s_L = 0.5 + 0 = 0.5 (v_internal + t_internal_to_root)
          s_R = 0 + 2 = 2 (v_C + t_C)
          logL_root = -0.5 * 1 * log(2*pi*1.0*(0.5+2)) - 0.5 * (1.0-1.0)^2 / (1.0*2.5)
                    = -0.5 * log(5*pi) - 0

        Total = -0.5*log(4*pi) - 1.0 - 0.5*log(5*pi)
        """
        a = TreeNode(id=0, name="A", branch_length=1.0)
        b = TreeNode(id=1, name="B", branch_length=1.0)
        internal = TreeNode(id=3, children=[a, b], branch_length=0.0)
        a.parent = internal
        b.parent = internal
        c = TreeNode(id=2, name="C", branch_length=2.0)
        root = TreeNode(id=4, children=[internal, c])
        internal.parent = root
        c.parent = root
        tree = Tree(root=root)

        data = {"A": np.array([0.0]), "B": np.array([2.0]), "C": np.array([1.0])}
        bm = BMLikelihood()
        logL = bm.log_likelihood(tree, data, sigma2=1.0)

        expected = -0.5 * np.log(4 * np.pi) - 1.0 - 0.5 * np.log(5 * np.pi)
        assert abs(logL - expected) < 1e-10, f"Expected {expected}, got {logL}"

    def test_vectorized_matches_loop(self):
        """Multi-dim likelihood should equal sum of per-dim likelihoods."""
        rng = np.random.RandomState(42)
        tree = random_tree(["A", "B", "C", "D", "E"], seed=42)
        D = 64
        data = {name: rng.randn(D) for name in tree.leaf_names()}
        bm = BMLikelihood()
        sigma2 = 0.5

        logL_vectorized = bm.log_likelihood(tree, data, sigma2)

        # Compute per-dimension naively
        logL_loop = 0.0
        for d in range(D):
            data_1d = {name: np.array([vec[d]]) for name, vec in data.items()}
            logL_loop += bm.log_likelihood(tree, data_1d, sigma2)

        assert abs(logL_vectorized - logL_loop) < 1e-6

    def test_higher_sigma_higher_likelihood_for_spread_data(self):
        """If data is spread out, higher sigma2 should give higher likelihood."""
        tree = parse_newick("((A:1,B:1):1,C:1);")
        data = {"A": np.array([0.0]), "B": np.array([10.0]), "C": np.array([20.0])}
        bm = BMLikelihood()
        logL_low = bm.log_likelihood(tree, data, sigma2=0.01)
        logL_high = bm.log_likelihood(tree, data, sigma2=100.0)
        assert logL_high > logL_low


# ---------------------------------------------------------------------------
# TestStochasticNNI
# ---------------------------------------------------------------------------

class TestStochasticNNI:
    def test_preserves_leaf_set(self):
        tree = random_tree(["A", "B", "C", "D", "E"], seed=42)
        nni = StochasticNNI(seed=0)
        new_tree, log_hr = nni.propose(tree)
        assert set(new_tree.leaf_names()) == set(tree.leaf_names())

    def test_symmetric_hastings_ratio(self):
        tree = random_tree(["A", "B", "C", "D", "E"], seed=42)
        nni = StochasticNNI(seed=1)
        _, log_hr = nni.propose(tree)
        assert log_hr == 0.0

    def test_topology_changes(self):
        tree = random_tree([f"t{i}" for i in range(10)], seed=42)
        changed = 0
        for i in range(20):
            nni = StochasticNNI(seed=i)
            new_tree, _ = nni.propose(tree)
            if write_newick(new_tree) != write_newick(tree):
                changed += 1
        assert changed > 0


# ---------------------------------------------------------------------------
# TestSubtreePruneRegraft
# ---------------------------------------------------------------------------

class TestSubtreePruneRegraft:
    def test_preserves_leaf_set(self):
        tree = random_tree([f"t{i}" for i in range(10)], seed=42)
        spr = SubtreePruneRegraft(seed=0)
        new_tree, _ = spr.propose(tree)
        assert set(new_tree.leaf_names()) == set(tree.leaf_names())

    def test_binary_preserved(self):
        tree = random_tree([f"t{i}" for i in range(10)], seed=42)
        spr = SubtreePruneRegraft(seed=1)
        new_tree, _ = spr.propose(tree)
        for node in new_tree.internals:
            assert len(node.children) == 2

    def test_topology_changes(self):
        tree = random_tree([f"t{i}" for i in range(10)], seed=42)
        changed = 0
        for i in range(30):
            spr = SubtreePruneRegraft(seed=i)
            new_tree, _ = spr.propose(tree)
            if write_newick(new_tree) != write_newick(tree):
                changed += 1
        assert changed > 0


# ---------------------------------------------------------------------------
# TestBranchLengthMultiplier
# ---------------------------------------------------------------------------

class TestBranchLengthMultiplier:
    def test_positive_branch_length(self):
        tree = random_tree(["A", "B", "C"], seed=42)
        prop = BranchLengthMultiplier(seed=0)
        for _ in range(50):
            new_tree, _ = prop.propose(tree)
            for n in new_tree.nodes:
                if not n.is_root():
                    assert n.branch_length > 0

    def test_hastings_ratio_nonzero(self):
        tree = random_tree(["A", "B", "C"], seed=42)
        prop = BranchLengthMultiplier(seed=0)
        _, log_hr = prop.propose(tree)
        assert log_hr != 0.0


# ---------------------------------------------------------------------------
# TestTreeLengthMultiplier
# ---------------------------------------------------------------------------

class TestTreeLengthMultiplier:
    def test_scales_all_branches(self):
        tree = random_tree(["A", "B", "C", "D"], seed=42)
        prop = TreeLengthMultiplier(seed=0, lambda_=0.1)
        old_total = tree.total_branch_length()
        new_tree, _ = prop.propose(tree)
        new_total = new_tree.total_branch_length()
        assert new_total != old_total
        # All branches scale by same factor
        ratio = new_total / old_total
        for old_n, new_n in zip(tree.postorder(), new_tree.postorder()):
            if not old_n.is_root():
                assert abs(new_n.branch_length / old_n.branch_length - ratio) < 1e-10


# ---------------------------------------------------------------------------
# TestSigmaMultiplier
# ---------------------------------------------------------------------------

class TestSigmaMultiplier:
    def test_positive(self):
        prop = SigmaMultiplier(seed=0)
        for _ in range(50):
            new_sigma2, _ = prop.propose_sigma(1.0)
            assert new_sigma2 > 0

    def test_hastings_ratio(self):
        prop = SigmaMultiplier(seed=0)
        new_sigma2, log_hr = prop.propose_sigma(1.0)
        expected_hr = np.log(new_sigma2 / 1.0)
        assert abs(log_hr - expected_hr) < 1e-10


# ---------------------------------------------------------------------------
# TestProposalMixer
# ---------------------------------------------------------------------------

class TestProposalMixer:
    def test_weighted_selection(self):
        mixer = ProposalMixer(
            proposal_names=["nni", "bl", "sigma"],
            weights=[6.0, 9.0, 1.0], seed=42,
        )
        counts = {"nni": 0, "bl": 0, "sigma": 0}
        for _ in range(10000):
            counts[mixer.select()] += 1
        assert counts["bl"] > counts["nni"] > counts["sigma"]

    def test_acceptance_tracking(self):
        mixer = ProposalMixer(
            proposal_names=["nni", "bl"], weights=[1.0, 1.0], seed=42,
        )
        mixer.record_acceptance("nni", True)
        mixer.record_acceptance("nni", True)
        mixer.record_acceptance("nni", False)
        assert abs(mixer.acceptance_rate("nni") - 2.0 / 3.0) < 1e-10


# ---------------------------------------------------------------------------
# TestMCMCChain
# ---------------------------------------------------------------------------

class TestMCMCChain:
    def test_synthetic_convergence(self):
        """MCMC on synthetic 5-taxon BM data should improve logL."""
        true_tree = parse_newick("((A:0.5,B:0.5):0.3,(C:0.4,(D:0.2,E:0.2):0.2):0.3);")
        data = simulate_bm(true_tree, 1.0, 32, seed=42)
        chain = MCMCChain(
            data=data, start_tree=random_tree(list(data.keys()), seed=99),
            sigma2_init=1.0, n_generations=5000, sample_freq=100, seed=42,
        )
        chain.run()
        assert len(chain.sampled_trees) > 0
        assert len(chain.sampled_logL) > 0
        assert chain.sampled_logL[-1] > chain.sampled_logL[0]


# ---------------------------------------------------------------------------
# TestMC3Runner
# ---------------------------------------------------------------------------

class TestMC3Runner:
    def test_heated_chains(self):
        true_tree = parse_newick("((A:0.5,B:0.5):0.3,C:0.8);")
        data = simulate_bm(true_tree, 1.0, 16, seed=42)
        runner = MC3Runner(
            data=data, n_chains=2, n_generations=1000,
            sample_freq=100, swap_freq=50, delta=0.1,
            start_tree=random_tree(list(data.keys()), seed=0), seed=42,
        )
        runner.run()
        assert len(runner.cold_chain.sampled_trees) > 0


# ---------------------------------------------------------------------------
# TestMultiRunOrchestrator
# ---------------------------------------------------------------------------

class TestMultiRunOrchestrator:
    def test_two_runs(self):
        true_tree = parse_newick("((A:0.5,B:0.5):0.3,C:0.8);")
        data = simulate_bm(true_tree, 1.0, 16, seed=42)
        orch = MultiRunOrchestrator(
            data=data, n_runs=2, n_chains=2,
            n_generations=1000, sample_freq=100, seed=42,
        )
        results = orch.run()
        assert len(results) == 2
        for r in results:
            assert len(r["sampled_trees"]) > 0


# ---------------------------------------------------------------------------
# TestDiagnostics
# ---------------------------------------------------------------------------

class TestDiagnostics:
    def test_ess_iid(self):
        samples = np.random.RandomState(42).randn(1000)
        ess = Diagnostics.effective_sample_size(samples)
        assert ess > 800

    def test_ess_correlated(self):
        x = np.cumsum(np.random.RandomState(42).randn(1000))
        ess = Diagnostics.effective_sample_size(x)
        assert ess < 100

    def test_psrf_similar_chains(self):
        c1 = np.random.RandomState(42).randn(500)
        c2 = np.random.RandomState(43).randn(500)
        psrf = Diagnostics.psrf([c1, c2])
        assert psrf < 1.2

    def test_asdsf_identical(self):
        trees = ["((A,B),C);"] * 100
        assert Diagnostics.asdsf([trees, trees]) == 0.0

    def test_asdsf_different(self):
        t1 = ["((A,B),C);"] * 100
        t2 = ["((A,C),B);"] * 100
        assert Diagnostics.asdsf([t1, t2]) > 0

    def test_get_splits(self):
        tree = parse_newick("((A,B),(C,D));")
        splits = Diagnostics._get_splits(tree)
        assert len(splits) > 0
        assert frozenset({"A", "B"}) in splits or frozenset({"C", "D"}) in splits


# ---------------------------------------------------------------------------
# TestConsensusBuilder
# ---------------------------------------------------------------------------

class TestConsensusBuilder:
    def test_unanimous_topology(self):
        trees = ["((A,B),(C,D));"] * 50
        consensus = ConsensusBuilder.majority_rule(trees, burnin_frac=0.0)
        assert consensus.n_leaves == 4
        splits = Diagnostics._get_splits(consensus)
        expected = Diagnostics._get_splits(parse_newick("((A,B),(C,D));"))
        assert splits == expected

    def test_majority_wins(self):
        trees = ["((A,B),(C,D));"] * 70 + ["((A,C),(B,D));"] * 30
        consensus = ConsensusBuilder.majority_rule(trees, burnin_frac=0.0)
        splits = Diagnostics._get_splits(consensus)
        ab = frozenset({"A", "B"})
        cd = frozenset({"C", "D"})
        assert ab in splits or cd in splits


# ---------------------------------------------------------------------------
# TestRobinsonFoulds
# ---------------------------------------------------------------------------

class TestRobinsonFoulds:
    def test_identical(self):
        t1 = parse_newick("((A,B),(C,D));")
        t2 = parse_newick("((A,B),(C,D));")
        assert robinson_foulds(t1, t2) == 0

    def test_different(self):
        t1 = parse_newick("((A,B),(C,D));")
        t2 = parse_newick("((A,C),(B,D));")
        assert robinson_foulds(t1, t2) > 0

    def test_symmetric(self):
        t1 = parse_newick("((A,B),(C,D));")
        t2 = parse_newick("((A,C),(B,D));")
        assert robinson_foulds(t1, t2) == robinson_foulds(t2, t1)


# ---------------------------------------------------------------------------
# TestEstimateSigma2
# ---------------------------------------------------------------------------

class TestEstimateSigma2:
    def test_positive(self):
        tree = parse_newick("((A:1,B:1):1,C:1);")
        data = simulate_bm(tree, 2.0, 32, seed=42)
        s2 = estimate_sigma2(data, tree)
        assert s2 > 0

    def test_scales_with_true_sigma(self):
        tree = parse_newick("((A:1,B:1):1,C:1);")
        data_low = simulate_bm(tree, 0.1, 128, seed=42)
        data_high = simulate_bm(tree, 10.0, 128, seed=42)
        s2_low = estimate_sigma2(data_low, tree)
        s2_high = estimate_sigma2(data_high, tree)
        assert s2_high > s2_low


# ---------------------------------------------------------------------------
# TestAutoTuning
# ---------------------------------------------------------------------------

class TestAutoTuning:
    def test_lambda_decreases_on_low_acceptance(self):
        prop = BranchLengthMultiplier(seed=0, lambda_=2.0)
        old_lambda = prop.lambda_
        prop.tune(0.05, batch=0)  # very low acceptance
        assert prop.lambda_ < old_lambda

    def test_lambda_increases_on_high_acceptance(self):
        prop = BranchLengthMultiplier(seed=0, lambda_=2.0)
        old_lambda = prop.lambda_
        prop.tune(0.80, batch=0)  # very high acceptance
        assert prop.lambda_ > old_lambda


# ---------------------------------------------------------------------------
# TestConsensusBranchLengths
# ---------------------------------------------------------------------------

class TestConsensusBranchLengths:
    def test_median_branch_lengths(self):
        """Consensus should use median branch lengths from posterior, not hardcoded."""
        trees = ["((A:0.5,B:0.3):0.2,C:0.7);"] * 50
        consensus = ConsensusBuilder.majority_rule(trees, burnin_frac=0.0)
        leaf_a = [n for n in consensus.leaves if n.name == "A"][0]
        assert abs(leaf_a.branch_length - 0.5) < 0.1, (
            f"Expected ~0.5, got {leaf_a.branch_length}"
        )

    def test_varied_branch_lengths_uses_median(self):
        """When branch lengths vary across trees, consensus should use median."""
        trees_short = ["((A:0.1,B:0.3):0.2,C:0.7);"] * 80
        trees_long = ["((A:10.0,B:0.3):0.2,C:0.7);"] * 20
        consensus = ConsensusBuilder.majority_rule(
            trees_short + trees_long, burnin_frac=0.0
        )
        leaf_a = [n for n in consensus.leaves if n.name == "A"][0]
        assert abs(leaf_a.branch_length - 0.1) < 0.05


# ---------------------------------------------------------------------------
# TestBranchBounds
# ---------------------------------------------------------------------------

class TestBranchBounds:
    def test_clamp_below_min(self):
        """clamp_branch_length should enforce minimum."""
        assert clamp_branch_length(1e-20) == BL_MIN
        assert clamp_branch_length(0.0) == BL_MIN

    def test_clamp_above_max(self):
        """clamp_branch_length should enforce maximum."""
        assert clamp_branch_length(200.0) == BL_MAX
        assert clamp_branch_length(1e10) == BL_MAX

    def test_clamp_in_range(self):
        """Values in range should pass through unchanged."""
        assert clamp_branch_length(0.5) == 0.5
        assert clamp_branch_length(1.0) == 1.0

    def test_bl_multiplier_respects_bounds(self):
        """BranchLengthMultiplier should clamp to [BL_MIN, BL_MAX]."""
        tree = parse_newick("((A:0.0000001,B:0.001):0.0001,C:0.001);")
        for i in range(100):
            prop = BranchLengthMultiplier(seed=i, lambda_=10.0)
            new_tree, _ = prop.propose(tree)
            for n in new_tree.nodes:
                if not n.is_root():
                    assert n.branch_length >= BL_MIN
                    assert n.branch_length <= BL_MAX

    def test_tl_multiplier_respects_bounds(self):
        """TreeLengthMultiplier should clamp to [BL_MIN, BL_MAX]."""
        tree = parse_newick("((A:50.0,B:80.0):30.0,C:90.0);")
        for i in range(100):
            prop = TreeLengthMultiplier(seed=i, lambda_=10.0)
            new_tree, _ = prop.propose(tree)
            for n in new_tree.nodes:
                if not n.is_root():
                    assert n.branch_length >= BL_MIN
                    assert n.branch_length <= BL_MAX


# ---------------------------------------------------------------------------
# TestNodeSlider
# ---------------------------------------------------------------------------

class TestNodeSlider:
    def test_preserves_leaf_set(self):
        """NodeSlider should not change leaf set."""
        tree = random_tree([f"t{i}" for i in range(8)], seed=42)
        slider = NodeSlider(seed=0)
        new_tree, _ = slider.propose(tree)
        assert set(new_tree.leaf_names()) == set(tree.leaf_names())

    def test_preserves_topology(self):
        """NodeSlider only changes branch lengths, not topology."""
        tree = random_tree([f"t{i}" for i in range(8)], seed=42)
        slider = NodeSlider(seed=0)
        old_splits = Diagnostics._get_splits(tree)
        new_tree, _ = slider.propose(tree)
        new_splits = Diagnostics._get_splits(new_tree)
        assert old_splits == new_splits

    def test_changes_branch_lengths(self):
        """NodeSlider should change at least some branch lengths."""
        tree = random_tree([f"t{i}" for i in range(8)], seed=42)
        changed = False
        for i in range(20):
            slider = NodeSlider(seed=i)
            new_tree, _ = slider.propose(tree)
            if abs(new_tree.total_branch_length() - tree.total_branch_length()) > 1e-10:
                changed = True
                break
        assert changed

    def test_positive_branch_lengths(self):
        """All proposed branch lengths should be positive and within bounds."""
        tree = random_tree([f"t{i}" for i in range(8)], seed=42)
        for i in range(50):
            slider = NodeSlider(seed=i)
            new_tree, _ = slider.propose(tree)
            for n in new_tree.nodes:
                if not n.is_root():
                    assert n.branch_length >= BL_MIN


# ---------------------------------------------------------------------------
# TestConfigurablePrior
# ---------------------------------------------------------------------------

class TestConfigurablePrior:
    def test_exp1_different_from_exp10(self):
        """Exp(1) and Exp(10) priors should give different log-prior values."""
        tree = parse_newick("((A:0.5,B:0.5):0.3,C:0.8);")
        lp_1 = MCMCChain._compute_log_prior(tree, 1.0, bl_prior_rate=1.0)
        lp_10 = MCMCChain._compute_log_prior(tree, 1.0, bl_prior_rate=10.0)
        assert lp_1 != lp_10
        # Exp(1) should be less penalizing for longer branches
        assert lp_1 > lp_10

    def test_chain_accepts_bl_prior_rate(self):
        """MCMCChain should accept bl_prior_rate parameter."""
        tree = parse_newick("((A:0.5,B:0.5):0.3,C:0.8);")
        data = simulate_bm(tree, 1.0, 16, seed=42)
        chain = MCMCChain(
            data=data, start_tree=tree.copy(),
            sigma2_init=1.0, n_generations=100, sample_freq=50,
            seed=42, bl_prior_rate=1.0,
        )
        chain.run()
        assert len(chain.sampled_trees) > 0


# ---------------------------------------------------------------------------
# TestPerResidueMode
# ---------------------------------------------------------------------------

class TestPerResidueMode:
    def test_load_msa(self):
        """load_msa should parse aligned FASTA into {name: sequence} dict."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(">A\nACDE-FG\n>B\nAC-EEFG\n>C\nACDEEFG\n")
            tmp_path = f.name
        try:
            msa = load_msa(tmp_path)
            assert len(msa) == 3
            assert len(msa["A"]) == 7
            assert msa["A"][4] == "-"
        finally:
            os.unlink(tmp_path)

    def test_load_msa_strips_pipe_suffix(self):
        """load_msa should strip |orgXXX suffixes from names."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as f:
            f.write(">P12345|org999\nACDEFG\n>Q67890|org123\nACDEFG\n")
            tmp_path = f.name
        try:
            msa = load_msa(tmp_path)
            assert "P12345" in msa
            assert "Q67890" in msa
        finally:
            os.unlink(tmp_path)

    def test_extract_aligned_embeddings(self):
        """extract_aligned_embeddings should map embeddings to aligned positions."""
        msa = {"A": "ACDE-FG", "B": "AC-EEFG"}
        rng = np.random.RandomState(42)
        embeddings = {
            "A": rng.randn(6, 4).astype(np.float32),
            "B": rng.randn(6, 4).astype(np.float32),
        }
        aligned = extract_aligned_embeddings(msa, embeddings, fill_value=0.0)
        assert aligned["A"].shape == (7, 4)
        assert aligned["B"].shape == (7, 4)
        # Gap positions should be fill_value
        assert np.all(aligned["A"][4] == 0.0)
        assert np.all(aligned["B"][2] == 0.0)
        # Non-gap positions should have actual values (not zero)
        assert not np.all(aligned["A"][0] == 0.0)

    def test_bm_with_higher_dim_data(self):
        """BM likelihood should work with higher-dimensional per-residue data."""
        tree = parse_newick("((A:0.5,B:0.5):0.3,C:0.8);")
        rng = np.random.default_rng(42)
        data = {
            "A": rng.standard_normal(20),
            "B": rng.standard_normal(20),
            "C": rng.standard_normal(20),
        }
        bm = BMLikelihood()
        logL = bm.log_likelihood(tree, data, sigma2=1.0)
        assert np.isfinite(logL)


# ---------------------------------------------------------------------------
# TestLabelBasedEvaluation
# ---------------------------------------------------------------------------

class TestLabelBasedEvaluation:
    """Tests for label-based tree evaluation metrics."""

    def _make_perfect_tree(self):
        """Tree where families are monophyletic: ((A1,A2),(B1,B2))."""
        return parse_newick("((A1:0.1,A2:0.1):0.5,(B1:0.1,B2:0.1):0.5);")

    def _make_mixed_tree(self):
        """Tree where families are NOT monophyletic: ((A1,B1),(A2,B2))."""
        return parse_newick("((A1:0.1,B1:0.1):0.5,(A2:0.1,B2:0.1):0.5);")

    def test_monophyly_perfect(self):
        """Perfect tree should have all families monophyletic."""
        tree = self._make_perfect_tree()
        labels = {"A1": "famA", "A2": "famA", "B1": "famB", "B2": "famB"}
        result = evaluate_monophyly(tree, labels)
        assert result["famA"]["monophyletic"] is True
        assert result["famB"]["monophyletic"] is True

    def test_monophyly_mixed(self):
        """Mixed tree should NOT have monophyletic families."""
        tree = self._make_mixed_tree()
        labels = {"A1": "famA", "A2": "famA", "B1": "famB", "B2": "famB"}
        result = evaluate_monophyly(tree, labels)
        assert result["famA"]["monophyletic"] is False
        assert result["famB"]["monophyletic"] is False

    def test_clade_purity_perfect(self):
        """Perfect tree should have high clade purity."""
        tree = self._make_perfect_tree()
        labels = {"A1": "famA", "A2": "famA", "B1": "famB", "B2": "famB"}
        purity = clade_purity_score(tree, labels)
        assert purity > 0.9

    def test_clade_purity_mixed(self):
        """Mixed tree should have lower clade purity."""
        tree = self._make_mixed_tree()
        labels = {"A1": "famA", "A2": "famA", "B1": "famB", "B2": "famB"}
        purity = clade_purity_score(tree, labels)
        assert purity < 0.8  # each internal node has 50% purity

    def test_family_separation(self):
        """Well-separated data should have high separation ratio."""
        rng = np.random.default_rng(42)
        data = {
            "A1": rng.standard_normal(10) + 10,
            "A2": rng.standard_normal(10) + 10,
            "B1": rng.standard_normal(10) - 10,
            "B2": rng.standard_normal(10) - 10,
        }
        labels = {"A1": "famA", "A2": "famA", "B1": "famB", "B2": "famB"}
        sep = family_separation_score(data, labels)
        assert sep["separation_ratio"] > 2.0
        assert sep["silhouette_approx"] > 0.5
