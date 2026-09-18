# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the ``load_response_dataset`` and
``load_preference_dataset`` dispatchers plus the shared
``resolve_external_dataset_class`` helper in
``nemo_rl.data.datasets.utils`` — covers both built-in registry lookup
and dotted-path import of user-defined dataset classes (see GitHub
issue #1020).
"""

from __future__ import annotations

import functools
import sys
import types
import warnings

import pytest

from nemo_rl.data.datasets.preference_datasets import (
    DATASET_REGISTRY as PREFERENCE_REGISTRY,
)
from nemo_rl.data.datasets.preference_datasets import (
    load_preference_dataset,
)
from nemo_rl.data.datasets.response_datasets import (
    DATASET_REGISTRY as RESPONSE_REGISTRY,
)
from nemo_rl.data.datasets.response_datasets import (
    load_response_dataset,
)
from nemo_rl.data.datasets.preference_datasets import (
    HelpSteer3Dataset as PreferenceHelpSteer3Dataset,
)
from nemo_rl.data.datasets.preference_datasets import (
    PreferenceDataset,
    Tulu3PreferenceDataset,
)
from nemo_rl.data.datasets.response_datasets import (
    AIMEDataset,
    GSM8KDataset,
    OpenMathInstruct2Dataset,
    ResponseDataset,
)
from nemo_rl.data.datasets.utils import (
    resolve_dataset_class,
    resolve_external_dataset_class,
    update_single_dataset_config,
    warn_on_unsupported_dataset_config_keys,
)


class _StubResponseDataset:
    """Minimal stub satisfying the contract used by
    ``load_response_dataset`` (constructor + ``set_task_spec`` +
    ``set_processor``)."""

    last_init_kwargs: dict | None = None

    def __init__(self, **kwargs):
        type(self).last_init_kwargs = kwargs
        self.task_spec_config = None
        self.processor_set = False

    def set_task_spec(self, data_config):
        self.task_spec_config = data_config

    def set_processor(self):
        self.processor_set = True


class _StubPreferenceDataset:
    """Minimal stub satisfying the contract used by
    ``load_preference_dataset`` (constructor + ``set_task_spec``)."""

    last_init_kwargs: dict | None = None

    def __init__(self, **kwargs):
        type(self).last_init_kwargs = kwargs
        self.task_spec_config = None

    def set_task_spec(self, data_config):
        self.task_spec_config = data_config


@pytest.fixture
def stub_module():
    """Install a throwaway module exposing the stub datasets for the
    duration of one test, then remove it from ``sys.modules``."""
    module_name = "nemo_rl_test_stub_external_module_for_1020"
    module = types.ModuleType(module_name)
    module.StubResponseDataset = _StubResponseDataset
    module.StubPreferenceDataset = _StubPreferenceDataset
    module.not_a_class = 42
    sys.modules[module_name] = module
    try:
        yield module_name
    finally:
        sys.modules.pop(module_name, None)


# ---------------------------------------------------------------------------
# resolve_external_dataset_class (shared helper in nemo_rl.data.datasets.utils)
# ---------------------------------------------------------------------------


def test_resolve_external_returns_class(stub_module):
    cls = resolve_external_dataset_class(f"{stub_module}.StubResponseDataset")
    assert cls is _StubResponseDataset


def test_resolve_external_rejects_unimportable_module():
    with pytest.raises(ValueError, match="Could not import module"):
        resolve_external_dataset_class("nemo_rl_does_not_exist_xyz.SomeDataset")


def test_resolve_external_rejects_missing_attribute(stub_module):
    with pytest.raises(ValueError, match="has no attribute 'Missing'"):
        resolve_external_dataset_class(f"{stub_module}.Missing")


def test_resolve_external_rejects_non_class(stub_module):
    with pytest.raises(ValueError, match="which is not a class"):
        resolve_external_dataset_class(f"{stub_module}.not_a_class")


# ---------------------------------------------------------------------------
# load_response_dataset
# ---------------------------------------------------------------------------


def test_load_response_dataset_builtin_registry_present():
    """Built-in registry must still expose the loadable formats."""
    assert "ResponseDataset" in RESPONSE_REGISTRY


def test_load_response_dataset_uses_external_class(stub_module):
    config = {
        "dataset_name": f"{stub_module}.StubResponseDataset",
        "data_path": "/tmp/does-not-need-to-exist.jsonl",
    }
    dataset = load_response_dataset(config)

    assert isinstance(dataset, _StubResponseDataset)
    assert _StubResponseDataset.last_init_kwargs == config
    # Lifecycle methods are exercised by the dispatcher.
    assert dataset.task_spec_config == config
    assert dataset.processor_set is True


def test_load_response_dataset_unknown_bare_name_errors():
    config = {"dataset_name": "definitely_not_in_registry"}
    with pytest.raises(ValueError, match="Unsupported dataset_name"):
        load_response_dataset(config)


def test_load_response_dataset_bad_dotted_path_errors():
    config = {"dataset_name": "nemo_rl_missing_module.MyDataset"}
    with pytest.raises(ValueError, match="Could not import module"):
        load_response_dataset(config)


# ---------------------------------------------------------------------------
# load_preference_dataset
# ---------------------------------------------------------------------------


def test_load_preference_dataset_builtin_registry_present():
    """Built-in registry must still expose the loadable formats."""
    assert "PreferenceDataset" in PREFERENCE_REGISTRY
    assert "BinaryPreferenceDataset" in PREFERENCE_REGISTRY


def test_load_preference_dataset_uses_external_class(stub_module):
    config = {
        "dataset_name": f"{stub_module}.StubPreferenceDataset",
        "data_path": "/tmp/does-not-need-to-exist.jsonl",
    }
    dataset = load_preference_dataset(config)

    assert isinstance(dataset, _StubPreferenceDataset)
    assert _StubPreferenceDataset.last_init_kwargs == config
    assert dataset.task_spec_config == config


def test_load_preference_dataset_unknown_bare_name_errors():
    config = {"dataset_name": "definitely_not_in_registry"}
    with pytest.raises(ValueError, match="Unsupported dataset_name"):
        load_preference_dataset(config)


def test_load_preference_dataset_bad_dotted_path_errors():
    config = {"dataset_name": "nemo_rl_missing_module.MyDataset"}
    with pytest.raises(ValueError, match="Could not import module"):
        load_preference_dataset(config)


# ---------------------------------------------------------------------------
# warn_on_unsupported_dataset_config_keys (dispatcher guard, issue #3270)
# ---------------------------------------------------------------------------


class _SplitOnlyDataset:
    """Stub declaring only ``split``; other behavioral keys are unsupported."""

    def __init__(self, split="train", **kwargs):
        pass


class _KwargsForwardingDataset(_SplitOnlyDataset):
    """Consumes keys via ``**kwargs`` and forwards to a base that declares
    them (mirrors the intent datasets)."""

    def __init__(self, **kwargs):
        kwargs.setdefault("split", "train")
        super().__init__(**kwargs)


def test_warn_on_unsupported_subset():
    with pytest.warns(UserWarning, match="subset='socratic'.*_SplitOnlyDataset"):
        warn_on_unsupported_dataset_config_keys(
            _SplitOnlyDataset, {"subset": "socratic"}
        )


def test_warn_on_unsupported_split_validation_size_and_seed():
    with pytest.warns(UserWarning) as record:
        warn_on_unsupported_dataset_config_keys(
            _SplitOnlyDataset, {"split_validation_size": 0.05, "seed": 42}
        )
    messages = [str(w.message) for w in record]
    assert any("split_validation_size=0.05" in m for m in messages)
    assert any("seed=42" in m for m in messages)


def test_no_warning_for_supported_key():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_on_unsupported_dataset_config_keys(_SplitOnlyDataset, {"split": "test"})


def test_no_warning_for_none_or_zero_values():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_on_unsupported_dataset_config_keys(
            _SplitOnlyDataset, {"subset": None, "split_validation_size": 0.0}
        )


def test_no_warning_when_base_class_declares_key():
    """MRO-aware: subclasses forwarding **kwargs to a declaring base must
    not be flagged."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_on_unsupported_dataset_config_keys(
            _KwargsForwardingDataset, {"split": "validation"}
        )


def test_warns_through_functools_partial():
    """Registry entries like the AIME variants are ``functools.partial``."""
    wrapped = functools.partial(_SplitOnlyDataset, split="train")
    with pytest.warns(UserWarning, match="_SplitOnlyDataset"):
        warn_on_unsupported_dataset_config_keys(wrapped, {"subset": "all"})


def test_load_response_dataset_warns_on_swallowed_key(monkeypatch, stub_module):
    """End-to-end: the dispatcher warns before instantiating a class that
    would silently swallow a behavioral key."""
    config = {
        "dataset_name": f"{stub_module}.StubResponseDataset",
        "subset": "socratic",
    }
    with pytest.warns(UserWarning, match="subset='socratic'"):
        load_response_dataset(config)


def test_warn_on_unsupported_data_path_and_input_key():
    """Constructor-consumed keys beyond the HF-loading four are checked too
    (e.g. `data_path` on a loader that hardcodes its source)."""
    with pytest.warns(UserWarning) as record:
        warn_on_unsupported_dataset_config_keys(
            _SplitOnlyDataset, {"data_path": "/tmp/my.jsonl", "input_key": "question"}
        )
    messages = [str(w.message) for w in record]
    assert any("data_path='/tmp/my.jsonl'" in m for m in messages)
    assert any("input_key='question'" in m for m in messages)


def test_load_preference_dataset_warns_on_swallowed_key(stub_module):
    """The preference dispatcher applies the same guard as the response one."""
    config = {
        "dataset_name": f"{stub_module}.StubPreferenceDataset",
        "split": "test",
    }
    with pytest.warns(UserWarning, match="split='test'"):
        load_preference_dataset(config)


# ---------------------------------------------------------------------------
# dataset_cls: canonical key, with dataset_name kept as the legacy alias
# ---------------------------------------------------------------------------

_LOADERS = [
    pytest.param(
        load_response_dataset,
        RESPONSE_REGISTRY,
        _StubResponseDataset,
        "StubResponseDataset",
        id="response",
    ),
    pytest.param(
        load_preference_dataset,
        PREFERENCE_REGISTRY,
        _StubPreferenceDataset,
        "StubPreferenceDataset",
        id="preference",
    ),
]


@pytest.mark.parametrize(("loader", "registry", "stub_cls", "stub_attr"), _LOADERS)
@pytest.mark.parametrize(
    "key_kind",
    ["dataset_cls_dotted_path", "dataset_cls_class_name", "dataset_name_legacy"],
)
def test_load_dataset_by_key(
    monkeypatch, stub_module, loader, registry, stub_cls, stub_attr, key_kind
):
    """``dataset_cls`` takes a built-in class name or a dotted path; a
    legacy ``dataset_name`` config loads exactly as before, with no warning."""
    monkeypatch.setitem(registry, "stub-id", stub_cls)
    if key_kind == "dataset_cls_dotted_path":
        config = {"dataset_cls": f"{stub_module}.{stub_attr}"}
    elif key_kind == "dataset_cls_class_name":
        # Looked up by the class name, not by the registry id "stub-id".
        config = {"dataset_cls": stub_cls.__name__}
    else:
        config = {"dataset_name": "stub-id"}

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dataset = loader(config)

    assert isinstance(dataset, stub_cls)
    assert stub_cls.last_init_kwargs == config


def test_dataset_cls_class_name_ignores_partial_bound_kwargs(monkeypatch):
    """Registry entries like the AIME variants are ``functools.partial``.
    ``dataset_cls`` names the wrapped class; options such as ``variant`` are
    ordinary config keys, not the partial's pre-bound ones."""
    monkeypatch.setitem(
        RESPONSE_REGISTRY,
        "stub-2025",
        functools.partial(_StubResponseDataset, variant="2025"),
    )
    load_response_dataset({"dataset_cls": "_StubResponseDataset"})
    assert _StubResponseDataset.last_init_kwargs == {
        "dataset_cls": "_StubResponseDataset"
    }

    load_response_dataset({"dataset_cls": "_StubResponseDataset", "variant": "2026"})
    assert _StubResponseDataset.last_init_kwargs["variant"] == "2026"


@pytest.mark.parametrize(("loader", "registry", "stub_cls", "stub_attr"), _LOADERS)
def test_dataset_cls_wins_over_dataset_name_with_warning(
    stub_module, loader, registry, stub_cls, stub_attr
):
    """Recipes inherit ``dataset_name`` from their exemplar, so both keys
    often arrive together. ``dataset_cls`` wins and the legacy value is
    never resolved (this one is not even importable)."""
    config = {
        "dataset_cls": f"{stub_module}.{stub_attr}",
        "dataset_name": "nemo_rl_missing_module.OldDataset",
    }
    with pytest.warns(UserWarning, match="dataset_name=.*is ignored"):
        dataset = loader(config)
    assert isinstance(dataset, stub_cls)


@pytest.mark.parametrize(("loader", "registry", "stub_cls", "stub_attr"), _LOADERS)
@pytest.mark.parametrize(
    ("config", "match"),
    [
        pytest.param(
            {"dataset_cls": "OpenMathInstruct-2"},
            "Unsupported dataset_cls='OpenMathInstruct-2'",
            id="registry-id-is-not-a-class-name",
        ),
        pytest.param(
            {"dataset_cls": "nemo_rl_missing_module.MyDataset"},
            "Could not import module .*dataset_cls=",
            id="unimportable-dotted-path",
        ),
        pytest.param({}, "dataset_cls", id="no-key"),
        pytest.param(
            {"dataset_cls": None, "dataset_name": None}, "dataset_cls", id="both-none"
        ),
    ],
)
def test_dataset_cls_errors(loader, registry, stub_cls, stub_attr, config, match):
    with pytest.raises(ValueError, match=match):
        loader(config)


@pytest.mark.parametrize(
    ("registry", "dataset_cls", "expected"),
    [
        (RESPONSE_REGISTRY, "OpenMathInstruct2Dataset", OpenMathInstruct2Dataset),
        (RESPONSE_REGISTRY, "ResponseDataset", ResponseDataset),
        (RESPONSE_REGISTRY, "GSM8KDataset", GSM8KDataset),
        (RESPONSE_REGISTRY, "AIMEDataset", AIMEDataset),
        (PREFERENCE_REGISTRY, "Tulu3PreferenceDataset", Tulu3PreferenceDataset),
        (PREFERENCE_REGISTRY, "HelpSteer3Dataset", PreferenceHelpSteer3Dataset),
        (PREFERENCE_REGISTRY, "PreferenceDataset", PreferenceDataset),
    ],
)
def test_builtin_class_names_resolve(registry, dataset_cls, expected):
    """Real registries, no instantiation: every built-in is reachable by its
    class name, per registry (the two ``HelpSteer3Dataset`` differ)."""
    assert resolve_dataset_class({"dataset_cls": dataset_cls}, registry, "") is expected


@pytest.mark.parametrize("registry", [RESPONSE_REGISTRY, PREFERENCE_REGISTRY])
def test_registry_class_names_unique(registry):
    """Two different classes sharing a name would make one unreachable."""
    classes = set()
    for entry in registry.values():
        while isinstance(entry, functools.partial):
            entry = entry.func
        classes.add(entry)
    names = [cls.__name__ for cls in classes]
    assert len(names) == len(set(names))


@pytest.mark.parametrize(
    ("entry", "default", "expected"),
    [
        pytest.param(
            {"dataset_name": "B"},
            {"dataset_cls": "A", "prompt_key": "p"},
            {"dataset_name": "B", "prompt_key": "p"},
            id="entry-name-beats-default-cls",
        ),
        pytest.param(
            {"dataset_cls": "B"},
            {"dataset_name": "A", "prompt_key": "p"},
            {"dataset_cls": "B", "prompt_key": "p"},
            id="entry-cls-beats-default-name",
        ),
        pytest.param(
            {}, {"dataset_cls": "A"}, {"dataset_cls": "A"}, id="entry-has-neither"
        ),
        pytest.param(
            {"dataset_name": None},
            {"dataset_cls": "A"},
            {"dataset_name": None, "dataset_cls": "A"},
            id="entry-name-none",
        ),
    ],
)
def test_default_does_not_override_entry_dataset_key(entry, default, expected):
    """``data.default`` fills ``dataset_cls``/``dataset_name`` as one slot, so
    a default never overrides the dataset an entry picked."""
    update_single_dataset_config(entry, default)
    assert entry == expected
