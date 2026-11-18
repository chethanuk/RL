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

import json
import tempfile

import pytest

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data.datasets import load_preference_dataset
from nemo_rl.data.utils import setup_preference_data


def create_sample_data(
    is_binary, prompt_key="prompt", chosen_key="chosen", rejected_key="rejected"
):
    data = [
        {
            prompt_key: "What is 2+2?",
            chosen_key: "The answer is 4.",
            rejected_key: "I don't know.",
        },
        {
            prompt_key: "What is the capital of France?",
            chosen_key: "The capital of France is Paris.",
            rejected_key: "The capital of France is London.",
        },
    ]

    if not is_binary:
        data = [
            {
                "context": [{"role": "user", "content": item[prompt_key]}],
                "completions": [
                    {
                        "rank": 0,
                        "completion": [
                            {"role": "assistant", "content": item[chosen_key]}
                        ],
                    },
                    {
                        "rank": 1,
                        "completion": [
                            {"role": "assistant", "content": item[rejected_key]}
                        ],
                    },
                ],
            }
            for item in data
        ]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        data_path = f.name

    return data_path


@pytest.mark.parametrize(
    "prompt_key,chosen_key,rejected_key",
    [("prompt", "chosen", "rejected"), ("question", "answer1", "answer2")],
)
def test_binary_preference_dataset(prompt_key, chosen_key, rejected_key):
    # load the dataset
    data_path = create_sample_data(
        is_binary=True,
        prompt_key=prompt_key,
        chosen_key=chosen_key,
        rejected_key=rejected_key,
    )
    data_config = {
        "dataset_name": "BinaryPreferenceDataset",
        "data_path": data_path,
        "prompt_key": prompt_key,
        "chosen_key": chosen_key,
        "rejected_key": rejected_key,
    }
    dataset = load_preference_dataset(data_config)

    # check prompt, chosen and rejected keys
    assert dataset.prompt_key == prompt_key
    assert dataset.chosen_key == chosen_key
    assert dataset.rejected_key == rejected_key

    # check the first example
    first_example = dataset.dataset[0]

    # only contains context, completions and task_name
    assert len(first_example.keys()) == 3
    assert "context" in first_example
    assert "completions" in first_example
    assert "task_name" in first_example

    # check the content
    assert first_example["context"] == [{"role": "user", "content": "What is 2+2?"}]
    assert first_example["completions"] == [
        {
            "completion": [{"role": "assistant", "content": "The answer is 4."}],
            "rank": 0,
        },
        {"completion": [{"role": "assistant", "content": "I don't know."}], "rank": 1},
    ]


def test_preference_dataset():
    # load the dataset
    data_path = create_sample_data(is_binary=False)
    data_config = {
        "dataset_name": "PreferenceDataset",
        "data_path": data_path,
    }
    dataset = load_preference_dataset(data_config)

    # check the first example
    first_example = dataset.dataset[0]

    # only contains context, completions and task_name
    assert len(first_example.keys()) == 3
    assert "context" in first_example
    assert "completions" in first_example
    assert "task_name" in first_example

    # check the content
    assert first_example["context"] == [{"role": "user", "content": "What is 2+2?"}]
    assert first_example["completions"] == [
        {
            "completion": [{"role": "assistant", "content": "The answer is 4."}],
            "rank": 0,
        },
        {"completion": [{"role": "assistant", "content": "I don't know."}], "rank": 1},
    ]


@pytest.mark.parametrize("dataset_name", ["HelpSteer3", "Tulu3Preference"])
def test_build_in_dataset(dataset_name):
    # load the dataset
    data_config = {"dataset_name": dataset_name}
    dataset = load_preference_dataset(data_config)

    # check the first example
    first_example = dataset.dataset[0]

    # only contains context, completions and task_name
    assert len(first_example.keys()) == 3
    assert "context" in first_example
    assert "completions" in first_example
    assert first_example["task_name"] == dataset_name

    # check the content
    assert first_example["context"][-1]["role"] == "user"
    for i in range(2):
        assert first_example["completions"][i]["rank"] == i
        assert first_example["completions"][i]["completion"][0]["role"] == "assistant"

    if dataset_name == "HelpSteer3":
        assert first_example["context"][-1]["content"][:20] == 'At the "tasks_B = [t'
        assert (
            first_example["completions"][0]["completion"][0]["content"][:20]
            == "Yes, you are correct"
        )
        assert (
            first_example["completions"][1]["completion"][0]["content"][:20]
            == "Sure! Here's the upd"
        )

    elif dataset_name == "Tulu3Preference":
        assert first_example["context"][-1]["content"][:20] == "Your entire response"
        assert (
            first_example["completions"][0]["completion"][0]["content"][:20]
            == "what is the true hea"
        )
        assert (
            first_example["completions"][1]["completion"][0]["content"][:20]
            == "it's a bit tricky as"
        )


def _write_jsonl(path, rows):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return str(path)


def _binary_rows(n):
    return [
        {"prompt": f"q{i}", "chosen": f"good {i}", "rejected": f"bad {i}"}
        for i in range(n)
    ]


def _ranked_rows(n):
    return [
        {
            "context": [{"role": "user", "content": f"q{i}"}],
            "completions": [
                {"rank": 0, "completion": [{"role": "assistant", "content": "good"}]},
                {"rank": 1, "completion": [{"role": "assistant", "content": "bad"}]},
            ],
        }
        for i in range(n)
    ]


@pytest.fixture(scope="module")
def preference_tokenizer():
    return get_tokenizer({"name": "Qwen/Qwen3-0.6B"})


@pytest.fixture
def preference_files(tmp_path):
    return {
        "a": _write_jsonl(tmp_path / "a.jsonl", _binary_rows(2)),
        "b": _write_jsonl(tmp_path / "b.jsonl", _binary_rows(3)),
        "ranked": _write_jsonl(tmp_path / "ranked.jsonl", _ranked_rows(2)),
    }


BINARY_DEFAULT = {"dataset_name": "BinaryPreferenceDataset"}


@pytest.mark.parametrize(
    "data_config,expected_train_len,expected_val_lens",
    [
        pytest.param(
            {
                "train": {"data_path": "a"},
                "validation": {"data_path": "b"},
                "default": BINARY_DEFAULT,
            },
            2,
            {"default": 3},
            id="single_dict_train_and_validation",
        ),
        pytest.param(
            {
                "train": [{"data_path": "a"}, {"data_path": "b"}],
                "default": BINARY_DEFAULT,
            },
            5,
            {},
            id="train_list",
        ),
        pytest.param(
            # Same task_name twice: rows from both entries are kept and share
            # one processor slot, as setup_response_data does.
            {
                "train": [{"data_path": "a"}, {"data_path": "a"}],
                "default": BINARY_DEFAULT,
            },
            4,
            {},
            id="same_file_listed_twice",
        ),
        pytest.param(
            {
                "train": [{"data_path": "a"}, {"data_path": "b"}],
                "validation": [
                    {"data_path": "a"},
                    {"data_path": "ranked", "dataset_name": "PreferenceDataset"},
                ],
                "default": BINARY_DEFAULT,
            },
            5,
            {"default": 4},
            id="train_and_validation_lists_merge_into_default",
        ),
        pytest.param(
            {
                "train": [{"data_path": "a"}, {"data_path": "b"}],
                "default": BINARY_DEFAULT,
                "val_data_paths": {"first": "ranked", "second": "ranked"},
            },
            5,
            {"first": 2, "second": 2},
            id="legacy_val_data_paths_with_train_list",
        ),
        pytest.param(
            {
                "train": {"data_path": "a"},
                "validation": [{"data_path": "b"}],
                "default": BINARY_DEFAULT,
                "val_data_paths": {"legacy": "ranked"},
            },
            2,
            {"legacy": 2},
            id="legacy_val_data_paths_takes_precedence_over_validation",
        ),
        pytest.param(
            {
                "train": {"data_path": "a", "dataset_name": "BinaryPreferenceDataset"},
                "default": None,
            },
            2,
            {},
            id="null_default",
        ),
    ],
)
def test_setup_preference_data_loads_dataset_lists(
    preference_tokenizer,
    preference_files,
    data_config,
    expected_train_len,
    expected_val_lens,
):
    def resolve(cfg):
        return {**cfg, "data_path": preference_files[cfg["data_path"]]}

    data_config = {**data_config, "max_input_seq_length": 128}
    for split in ("train", "validation"):
        cfg = data_config.get(split)
        if isinstance(cfg, list):
            data_config[split] = [resolve(c) for c in cfg]
        elif cfg is not None:
            data_config[split] = resolve(cfg)
    if "val_data_paths" in data_config:
        data_config["val_data_paths"] = {
            name: preference_files[key]
            for name, key in data_config["val_data_paths"].items()
        }

    train, val = setup_preference_data(preference_tokenizer, data_config)

    assert len(train) == expected_train_len
    assert {name: len(ds) for name, ds in val.items()} == expected_val_lens
    # Every merged row must reach a processor for its own task and tokenize.
    for ds in [train, *val.values()]:
        for i in range(len(ds)):
            assert ds[i]["message_log_chosen"]
