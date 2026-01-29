"""Tests for the Language base class and shared functionality."""
import pytest
import numpy as np
import torch
import random
from tinylang.language import Language, AR, PCFG


# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)


class TestLanguageConstants:
    """Test that base class constants are inherited correctly."""

    def test_base_class_constants(self):
        """Test that Language has the expected constants."""
        assert Language.PAD == 0
        assert Language.BOS == 1
        assert Language.EOS == 2

    def test_ar_inherits_constants(self):
        """Test that AR inherits PAD/BOS/EOS from base class."""
        ar = AR(num_kv=20, max_length=20)
        assert ar.PAD == 0
        assert ar.BOS == 1
        assert ar.EOS == 2

    def test_pcfg_inherits_constants(self):
        """Test that PCFG inherits PAD/BOS/EOS from base class."""
        pcfg = PCFG(
            num_terminals=10,
            num_nonterminals=5,
            max_rhs_len=3,
            max_rules_per_nt=2,
            max_depth=3,
        )
        assert pcfg.PAD == 0
        assert pcfg.BOS == 1
        assert pcfg.EOS == 2


class TestARBatchify:
    """Test AR batchify functionality."""

    @pytest.fixture
    def ar(self):
        np.random.seed(42)
        # num_kv=20 gives 10 keys, max_length=20 means max 10 pairs, so this works
        return AR(num_kv=20, max_length=20, mask_nonquery=False)

    @pytest.fixture
    def ar_masked(self):
        np.random.seed(42)
        return AR(num_kv=20, max_length=20, mask_nonquery=True)

    def test_batchify_returns_expected_keys(self, ar):
        """Test that batchify returns dict with expected keys."""
        tok, schema = ar.sample()
        batch = ar.batchify([tok], [schema])
        assert "input_ids" in batch
        assert "labels" in batch
        assert "strs" in batch
        assert "probing_schemas" in batch

    def test_batchify_tensor_shapes(self, ar):
        """Test that batchify returns correctly shaped tensors."""
        toks, schemas = [], []
        for _ in range(4):
            tok, schema = ar.sample()
            toks.append(tok)
            schemas.append(schema)

        batch = ar.batchify(toks, schemas)
        assert batch["input_ids"].shape[0] == 4
        assert batch["labels"].shape[0] == 4
        assert len(batch["strs"]) == 4
        assert len(batch["probing_schemas"]) == 4

    def test_batchify_padding(self, ar):
        """Test that sequences are padded correctly."""
        # Generate samples of varying lengths
        toks, schemas = [], []
        for _ in range(3):
            tok, schema = ar.sample()
            toks.append(tok)
            schemas.append(schema)

        batch = ar.batchify(toks, schemas)
        # All sequences should have same length after padding
        assert batch["input_ids"].shape[1] == max(len(t) for t in toks)

    def test_batchify_labels_mask_pad(self, ar):
        """Test that PAD tokens are masked with -100 in labels."""
        toks, schemas = [], []
        for _ in range(3):
            tok, schema = ar.sample()
            toks.append(tok)
            schemas.append(schema)

        batch = ar.batchify(toks, schemas)
        # Where input is PAD (0), labels should be -100
        pad_positions = batch["input_ids"] == ar.PAD
        assert (batch["labels"][pad_positions] == -100).all()

    def test_batchify_mask_nonquery(self, ar_masked):
        """Test that mask_nonquery masks all but target position."""
        tok, schema = ar_masked.sample()
        batch = ar_masked.batchify([tok], [schema])

        target_pos = schema["queries"]["target_item"]["pos"]
        labels = batch["labels"][0]

        # Only target position should not be -100
        non_masked = (labels != -100).nonzero(as_tuple=True)[0]
        assert len(non_masked) == 1
        assert non_masked[0].item() == target_pos

    def test_batchify_no_strs_pretty(self, ar):
        """Test that AR batchify does not add strs_pretty (no override)."""
        tok, schema = ar.sample()
        batch = ar.batchify([tok], [schema], verbose=True)
        # AR doesn't override _batchify_extras, so no strs_pretty
        assert "strs_pretty" not in batch


class TestPCFGBatchify:
    """Test PCFG batchify functionality."""

    @pytest.fixture
    def pcfg(self):
        np.random.seed(42)
        return PCFG(
            num_terminals=10,
            num_nonterminals=5,
            max_rhs_len=3,
            max_rules_per_nt=2,
            max_depth=3,
            mask_nonquery=False,
        )

    @pytest.fixture
    def pcfg_masked(self):
        np.random.seed(42)
        return PCFG(
            num_terminals=10,
            num_nonterminals=5,
            max_rhs_len=3,
            max_rules_per_nt=2,
            max_depth=3,
            mask_nonquery=True,
        )

    def test_batchify_returns_expected_keys(self, pcfg):
        """Test that batchify returns dict with expected keys."""
        tok, schema = pcfg.sample()
        batch = pcfg.batchify([tok], [schema])
        assert "input_ids" in batch
        assert "labels" in batch
        assert "strs" in batch
        assert "probing_schemas" in batch

    def test_batchify_verbose_adds_strs_pretty(self, pcfg):
        """Test that PCFG adds strs_pretty when verbose=True."""
        tok, schema = pcfg.sample()
        batch = pcfg.batchify([tok], [schema], verbose=True)
        assert "strs_pretty" in batch
        assert len(batch["strs_pretty"]) == 1

    def test_batchify_not_verbose_no_strs_pretty(self, pcfg):
        """Test that PCFG does not add strs_pretty when verbose=False."""
        tok, schema = pcfg.sample()
        batch = pcfg.batchify([tok], [schema], verbose=False)
        assert "strs_pretty" not in batch

    def test_batchify_mask_nonquery(self, pcfg_masked):
        """Test that mask_nonquery masks all but target position."""
        tok, schema = pcfg_masked.sample()
        batch = pcfg_masked.batchify([tok], [schema])

        target_pos = schema["queries"]["target_item"]["pos"]
        labels = batch["labels"][0]

        # Only target position should not be -100
        non_masked = (labels != -100).nonzero(as_tuple=True)[0]
        assert len(non_masked) == 1
        assert non_masked[0].item() == target_pos


class TestGetTrainStep:
    """Test get_train_step for both AR and PCFG."""

    def test_ar_get_train_step(self):
        """Test AR get_train_step generates batches correctly."""
        np.random.seed(42)
        ar = AR(num_kv=20, max_length=20)
        batch = ar.get_train_step(step=0, batch_size=4)
        assert batch["input_ids"].shape[0] == 4

    def test_pcfg_get_train_step(self):
        """Test PCFG get_train_step generates batches correctly."""
        np.random.seed(42)
        pcfg = PCFG(
            num_terminals=10,
            num_nonterminals=5,
            max_rhs_len=3,
            max_rules_per_nt=2,
            max_depth=3,
        )
        batch = pcfg.get_train_step(step=0, batch_size=4)
        assert batch["input_ids"].shape[0] == 4


class TestPrepareSets:
    """Test prepare_sets for both AR and PCFG."""

    def test_ar_prepare_sets(self):
        """Test AR prepare_sets creates eval sets."""
        np.random.seed(42)
        ar = AR(num_kv=20, max_length=20)
        ar.prepare_sets(train_set_size=10, eval_set_size=5)
        assert "test" in ar.evalsets
        assert len(ar.evalsets["test"]["toks"]) == 5

    def test_pcfg_prepare_sets(self):
        """Test PCFG prepare_sets creates eval sets."""
        np.random.seed(42)
        pcfg = PCFG(
            num_terminals=10,
            num_nonterminals=5,
            max_rhs_len=3,
            max_rules_per_nt=2,
            max_depth=3,
        )
        pcfg.prepare_sets(train_set_size=10, eval_set_size=5)
        assert "test" in pcfg.evalsets
        assert len(pcfg.evalsets["test"]["toks"]) == 5

    def test_ar_prepare_sets_with_split(self):
        """Test AR prepare_sets creates dev set when train_test_split > 0."""
        np.random.seed(42)
        ar = AR(num_kv=20, max_length=20, train_test_split=0.2)
        ar.prepare_sets(train_set_size=10, eval_set_size=5)
        assert "dev" in ar.evalsets
        assert "test" in ar.evalsets

    def test_pcfg_prepare_sets_with_split(self):
        """Test PCFG prepare_sets creates dev set when train_test_split > 0."""
        np.random.seed(42)
        pcfg = PCFG(
            num_terminals=10,
            num_nonterminals=5,
            max_rhs_len=3,
            max_rules_per_nt=2,
            max_depth=3,
            train_test_split=0.2,
        )
        pcfg.prepare_sets(train_set_size=10, eval_set_size=5)
        assert "dev" in pcfg.evalsets
        assert "test" in pcfg.evalsets


class TestGetEvalStep:
    """Test get_eval_step for both AR and PCFG."""

    def test_ar_get_eval_step(self):
        """Test AR get_eval_step returns batches from eval set."""
        np.random.seed(42)
        ar = AR(num_kv=20, max_length=20)
        ar.prepare_sets(train_set_size=10, eval_set_size=8)
        batch = ar.get_eval_step(step=0, batch_size=4)
        assert batch["input_ids"].shape[0] == 4

    def test_pcfg_get_eval_step(self):
        """Test PCFG get_eval_step returns batches from eval set."""
        np.random.seed(42)
        pcfg = PCFG(
            num_terminals=10,
            num_nonterminals=5,
            max_rhs_len=3,
            max_rules_per_nt=2,
            max_depth=3,
        )
        pcfg.prepare_sets(train_set_size=10, eval_set_size=8)
        batch = pcfg.get_eval_step(step=0, batch_size=4)
        assert batch["input_ids"].shape[0] == 4

    def test_pcfg_get_eval_step_has_strs_pretty(self):
        """Test PCFG get_eval_step includes strs_pretty (verbose=True)."""
        np.random.seed(42)
        pcfg = PCFG(
            num_terminals=10,
            num_nonterminals=5,
            max_rhs_len=3,
            max_rules_per_nt=2,
            max_depth=3,
        )
        pcfg.prepare_sets(train_set_size=10, eval_set_size=8)
        batch = pcfg.get_eval_step(step=0, batch_size=4)
        assert "strs_pretty" in batch
