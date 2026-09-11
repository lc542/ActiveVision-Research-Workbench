
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from activevision_workbench import (
    AdapterLifecycleError,
    AdapterOutputError,
    AdapterRegistry,
    AdapterUnavailableError,
    ArtifactReference,
    Capability,
    CapabilitySet,
    ContractValidationError,
    DuplicateAdapterError,
    FixationEvent,
    InferenceRequest,
    ModelIdentity,
    PredictionRecord,
    RegistryError,
    RequestRequirements,
    SerializationError,
    StoppingReason,
    UnknownModelError,
    UnsupportedModelError,
    dumps_prediction,
    dumps_request,
    loads_prediction,
    loads_request,
    read_predictions_jsonl,
    read_requests_jsonl,
    write_predictions_jsonl,
    write_requests_jsonl,
)
from activevision_workbench.adapters import AdapterState
from activevision_workbench.adapters.dummy import (
    DUMMY_CAPABILITIES,
    DUMMY_IDENTITY,
    DUMMY_MODEL_ID,
    DUMMY_MODEL_VARIANT,
    DummyAdapter,
    create_dummy_registry,
    dummy_descriptor,
)
from activevision_workbench.registry import AdapterDescriptor


def make_request(**overrides: object) -> InferenceRequest:
    values: dict[str, object] = {
        "model_id": DUMMY_MODEL_ID,
        "model_variant": DUMMY_MODEL_VARIANT,
        "dataset_id": "fixture_dataset",
        "dataset_split": "test",
        "item_id": "image-001",
        "image_ref": "relative/images/image-001.png",
        "image_width": 640,
        "image_height": 480,
        "task_text": "find the vehicle",
        "observer_id": "observer-7",
        "observer_metadata": {"group": "fixture", "age": 30},
        "num_samples": 3,
        "base_seed": 41,
        "max_fixations": 3,
        "time_horizon_s": 1.0,
        "model_options": {
            "dummy": {"explanation_prefix": "Synthetic"},
        },
    }
    values.update(overrides)
    return InferenceRequest.create(**values)  # type: ignore[arg-type]


class CapabilityTests(unittest.TestCase):
    def test_required_vocabulary_and_runtime_query(self) -> None:
        self.assertEqual(
            {capability.value for capability in Capability},
            {
                "produces_scanpaths",
                "produces_saliency_maps",
                "stochastic_multi_sample",
                "instruction_conditioned",
                "observer_conditioned",
                "continuous_time_output",
                "fixation_duration_output",
                "text_explanation_output",
                "batch_inference",
            },
        )
        capabilities = CapabilitySet(
            frozenset(
                {
                    Capability.PRODUCES_SCANPATHS,
                    Capability.TEXT_EXPLANATION_OUTPUT,
                }
            )
        )
        self.assertTrue(capabilities.supports(Capability.PRODUCES_SCANPATHS))
        self.assertFalse(capabilities.supports(Capability.BATCH_INFERENCE))

    def test_capability_serialization_is_sorted_and_round_trips(self) -> None:
        data = DUMMY_CAPABILITIES.to_dict()
        self.assertEqual(data["values"], sorted(data["values"]))
        encoded = json.dumps(data, allow_nan=False, sort_keys=True)
        restored = CapabilitySet.from_dict(json.loads(encoded))
        self.assertEqual(restored, DUMMY_CAPABILITIES)

    def test_unknown_capability_is_rejected(self) -> None:
        with self.assertRaisesRegex(ContractValidationError, "unknown capability"):
            CapabilitySet.from_dict(
                {"schema_version": 1, "values": ["future_capability"]}
            )


class InferenceRequestTests(unittest.TestCase):
    def test_deterministic_id_is_stable_across_mapping_order(self) -> None:
        first = make_request(
            observer_metadata={"a": 1, "b": 2},
            model_options={"dummy": {"x": [1, 2], "y": "value"}},
        )
        second = make_request(
            observer_metadata={"b": 2, "a": 1},
            model_options={"dummy": {"y": "value", "x": [1, 2]}},
        )
        self.assertEqual(first.request_id, second.request_id)
        self.assertEqual(first.request_id, first.deterministic_id())
        self.assertTrue(first.request_id.startswith("req_"))
        self.assertEqual(len(first.request_id), 68)

    def test_deterministic_id_covers_scientific_inputs(self) -> None:
        baseline = make_request()
        changes = (
            {"item_id": "image-002"},
            {"task_text": "find the person"},
            {"observer_id": "observer-8", "observer_metadata": {}},
            {"base_seed": 42},
            {"num_samples": 2},
            {"model_options": {"dummy": {"explanation_prefix": "Other"}}},
        )
        for change in changes:
            with self.subTest(change=change):
                changed = make_request(**change)
                self.assertNotEqual(baseline.request_id, changed.request_id)

    def test_caller_supplied_id_is_preserved(self) -> None:
        request = make_request(request_id="external-request-id")
        self.assertEqual(request.request_id, "external-request-id")
        self.assertNotEqual(request.request_id, request.deterministic_id())

    def test_full_request_round_trip_and_immutability(self) -> None:
        request = make_request(target_description="vehicle")
        restored = loads_request(dumps_request(request))
        self.assertEqual(restored, request)
        self.assertEqual(restored.task_text, "find the vehicle")
        self.assertEqual(restored.observer_metadata["group"], "fixture")
        with self.assertRaises(TypeError):
            restored.observer_metadata["group"] = "changed"  # type: ignore[index]

    def test_image_reference_is_opaque(self) -> None:
        request = make_request(image_ref="does-not-exist://opaque/reference")
        self.assertEqual(request.image_ref, "does-not-exist://opaque/reference")

    def test_invalid_dimension_and_sample_types(self) -> None:
        cases = (
            ({"image_width": 0}, "image_width"),
            ({"image_width": True}, "image_width"),
            ({"image_height": None}, "image_height"),
            ({"num_samples": 0}, "num_samples"),
            ({"num_samples": True}, "num_samples"),
            ({"max_fixations": 0}, "max_fixations"),
            ({"time_horizon_s": 0.0}, "time_horizon_s"),
        )
        for change, field in cases:
            with self.subTest(change=change):
                with self.assertRaises(ContractValidationError) as caught:
                    make_request(**change)
                self.assertIn(field, caught.exception.field)

    def test_task_and_observer_are_not_coerced(self) -> None:
        for change, field in (
            ({"task_text": 17}, "task_text"),
            ({"observer_id": 17, "observer_metadata": {}}, "observer_id"),
        ):
            with self.subTest(change=change):
                with self.assertRaises(ContractValidationError) as caught:
                    make_request(**change)
                self.assertIn(field, caught.exception.field)

    def test_adapter_requirements_report_missing_task_and_observer(self) -> None:
        missing_task = make_request(task_text=None, target_description=None)
        with self.assertRaises(ContractValidationError) as caught_task:
            missing_task.validate_for(
                DUMMY_CAPABILITIES,
                RequestRequirements(instruction=True, observer=True),
            )
        self.assertEqual(caught_task.exception.field, "InferenceRequest.task_text")

        missing_observer = make_request(observer_id=None, observer_metadata={})
        with self.assertRaises(ContractValidationError) as caught_observer:
            missing_observer.validate_for(
                DUMMY_CAPABILITIES,
                RequestRequirements(instruction=True, observer=True),
            )
        self.assertEqual(
            caught_observer.exception.field,
            "InferenceRequest.observer_id",
        )

    def test_empty_task_text_is_explicit_not_missing(self) -> None:
        request = make_request(task_text="")
        request.validate_for(
            DUMMY_CAPABILITIES,
            RequestRequirements(instruction=True, observer=True),
        )
        self.assertEqual(request.task_text, "")

    def test_anonymous_observer_metadata_can_satisfy_requirement(self) -> None:
        request = make_request(
            observer_id=None,
            observer_metadata={"synthetic_observer": 3},
        )
        request.validate_for(
            DUMMY_CAPABILITIES,
            RequestRequirements(instruction=True, observer=True),
        )
        self.assertIsNone(request.observer_id)

    def test_capability_validation_rejects_unsupported_fields(self) -> None:
        only_scanpath = CapabilitySet(
            frozenset({Capability.PRODUCES_SCANPATHS})
        )
        with self.assertRaisesRegex(
            ContractValidationError,
            "instruction-conditioned",
        ):
            make_request(num_samples=1, time_horizon_s=None).validate_for(only_scanpath)
        request = make_request(
            task_text=None,
            target_description=None,
            observer_id=None,
            observer_metadata={},
            time_horizon_s=None,
        )
        with self.assertRaisesRegex(ContractValidationError, "multi-sample"):
            request.validate_for(only_scanpath)

    def test_model_options_are_namespaced_and_json_strict(self) -> None:
        cases = (
            {"temperature": {"value": 1}},
            {"dummy": 1},
            {"dummy": {"bad": {1, 2}}},
            {"dummy": {"bad": float("nan")}},
            {"dummy": {1: "non-string key"}},
        )
        for options in cases:
            with self.subTest(options=options):
                with self.assertRaises(ContractValidationError):
                    make_request(model_options=options)

    def test_unknown_fields_and_schema_versions_are_rejected(self) -> None:
        data = make_request().to_dict()
        data["future_field"] = "value"
        with self.assertRaisesRegex(ContractValidationError, "unknown field"):
            InferenceRequest.from_dict(data)
        data.pop("future_field")
        data["schema_version"] = 2
        with self.assertRaisesRegex(ContractValidationError, "unsupported schema"):
            InferenceRequest.from_dict(data)


class FixationAndPredictionTests(unittest.TestCase):
    def test_fixation_round_trip_preserves_native_timing(self) -> None:
        fixation = FixationEvent(
            x_px=12.5,
            y_px=24.5,
            x_norm=0.25,
            y_norm=0.5,
            sequence_index=0,
            timestamp_s=0.0,
            duration_s=0.0,
            confidence=4.2,
            native_timing={
                "value": 125,
                "unit": "ms",
                "meaning": "duration",
            },
        )
        self.assertEqual(FixationEvent.from_dict(fixation.to_dict()), fixation)

    def test_coordinate_validation(self) -> None:
        cases = (
            {"x_px": float("nan"), "y_px": 1.0},
            {"x_px": 1.0},
            {"x_norm": 0.2},
            {"x_norm": -0.01, "y_norm": 0.2},
            {"x_norm": 1.01, "y_norm": 0.2},
            {},
        )
        for values in cases:
            with self.subTest(values=values):
                with self.assertRaises(ContractValidationError):
                    FixationEvent(**values)
        for boundary in (0.0, 1.0):
            FixationEvent(x_norm=boundary, y_norm=boundary)

    def test_negative_and_nonfinite_timing_is_rejected(self) -> None:
        for values in (
            {"timestamp_s": -0.1},
            {"duration_s": -0.1},
            {"timestamp_s": float("inf")},
            {"duration_s": float("nan")},
        ):
            with self.subTest(values=values):
                with self.assertRaises(ContractValidationError):
                    FixationEvent(x_norm=0.2, y_norm=0.3, **values)

    def test_core_does_not_invent_timing_fields(self) -> None:
        timestamp_only = FixationEvent(
            x_norm=0.2,
            y_norm=0.3,
            timestamp_s=1.0,
        )
        duration_only = FixationEvent(
            x_norm=0.2,
            y_norm=0.3,
            duration_s=0.5,
        )
        self.assertIsNone(timestamp_only.duration_s)
        self.assertIsNone(duration_only.timestamp_s)

    def test_timestamp_ordering_rejects_regression(self) -> None:
        first = FixationEvent(x_norm=0.1, y_norm=0.1, timestamp_s=1.0)
        second = FixationEvent(x_norm=0.2, y_norm=0.2, timestamp_s=0.5)
        with self.assertRaises(ContractValidationError) as caught:
            self._record((first, second))
        self.assertIn("fixations[1].timestamp_s", caught.exception.field)

    def test_continuous_time_rejects_partial_timestamps(self) -> None:
        record = self._record(
            (
                FixationEvent(x_norm=0.1, y_norm=0.1, timestamp_s=0.0),
                FixationEvent(x_norm=0.2, y_norm=0.2),
            )
        )
        capabilities = CapabilitySet(
            frozenset(
                {
                    Capability.PRODUCES_SCANPATHS,
                    Capability.CONTINUOUS_TIME_OUTPUT,
                }
            )
        )
        with self.assertRaisesRegex(ContractValidationError, "cannot mix"):
            record.validate_for_capabilities(capabilities)

    def test_full_prediction_json_round_trip(self) -> None:
        fixation = FixationEvent(
            x_px=1.0,
            y_px=2.0,
            x_norm=0.1,
            y_norm=0.2,
            sequence_index=0,
            timestamp_s=0.0,
            duration_s=0.2,
            native_timing={"value": 200, "unit": "ms"},
        )
        record = self._record(
            (fixation,),
            saliency_artifact=ArtifactReference(
                uri="relative/saliency.npy",
                kind="saliency_map",
            ),
            explanation_text="Unicode explanation: café 视线",
            native_artifacts=(
                ArtifactReference(
                    uri="relative/native.json",
                    kind="model_native_prediction",
                    metadata={"format_version": 1},
                ),
            ),
        )
        restored = loads_prediction(dumps_prediction(record))
        self.assertEqual(restored, record)
        self.assertIsInstance(restored.fixations[0], FixationEvent)
        self.assertEqual(restored.native_artifacts[0].uri, "relative/native.json")
        self.assertEqual(restored.explanation_text, "Unicode explanation: café 视线")

    def test_prediction_compatibility_and_nested_error_paths(self) -> None:
        record = self._record(
            (FixationEvent(x_norm=0.1, y_norm=0.2, duration_s=0.2),)
        )
        data = record.to_dict()
        data["future_field"] = True
        with self.assertRaisesRegex(ContractValidationError, "unknown field"):
            PredictionRecord.from_dict(data)

        data = record.to_dict()
        data["schema_version"] = 2
        with self.assertRaisesRegex(ContractValidationError, "unsupported schema"):
            PredictionRecord.from_dict(data)

        data = record.to_dict()
        data["stopping_reason"] = "future_reason"
        with self.assertRaisesRegex(ContractValidationError, "future_reason"):
            PredictionRecord.from_dict(data)

        data = record.to_dict()
        data["fixations"][0]["duration_s"] = -1.0  # type: ignore[index]
        with self.assertRaises(ContractValidationError) as caught:
            PredictionRecord.from_dict(data)
        self.assertEqual(
            caught.exception.field,
            "PredictionRecord.fixations[0].duration_s",
        )

    @staticmethod
    def _record(
        fixations: tuple[FixationEvent, ...],
        **overrides: object,
    ) -> PredictionRecord:
        values: dict[str, object] = {
            "run_id": "run-1",
            "request_id": "request-1",
            "model": ModelIdentity(
                model_id="model",
                model_variant="variant",
                adapter_version="1",
            ),
            "dataset_id": "dataset",
            "dataset_split": "test",
            "item_id": "item",
            "sample_id": "sample-0",
            "sample_index": 0,
            "seed": 1,
            "observer_id": "observer",
            "task_text": "task",
            "fixations": fixations,
            "native_metadata": {"native": True},
            "stopping_reason": StoppingReason.COMPLETED,
        }
        values.update(overrides)
        return PredictionRecord(**values)  # type: ignore[arg-type]


class SerializationTests(unittest.TestCase):
    def test_three_samples_remain_three_jsonl_records(self) -> None:
        request = make_request(num_samples=3)
        with create_dummy_registry().create(DUMMY_MODEL_ID) as adapter:
            predictions = adapter.predict(request, run_id="run-3")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "predictions.jsonl")
            write_predictions_jsonl(path, predictions)
            restored = read_predictions_jsonl(path)
            self.assertEqual(len(restored), 3)
            self.assertEqual([record.sample_index for record in restored], [0, 1, 2])
            self.assertEqual(len({record.sample_id for record in restored}), 3)
            self.assertTrue(path.read_bytes().endswith(b"\n"))

    def test_request_jsonl_round_trip_and_empty_file(self) -> None:
        request = make_request(num_samples=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "requests.jsonl")
            write_requests_jsonl(path, (request,))
            self.assertEqual(read_requests_jsonl(path), (request,))
            empty = Path(directory, "empty.jsonl")
            empty.touch()
            self.assertEqual(read_requests_jsonl(empty), ())

    def test_write_refuses_overwrite_and_preserves_existing_file(self) -> None:
        request = make_request(num_samples=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "requests.jsonl")
            path.write_text("original\n", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                write_requests_jsonl(path, (request,))
            self.assertEqual(path.read_text(encoding="utf-8"), "original\n")

    def test_malformed_jsonl_reports_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "bad.jsonl")
            request = make_request(num_samples=1)
            with create_dummy_registry().create(DUMMY_MODEL_ID) as adapter:
                record = adapter.predict(request, run_id="run-line")[0]
            path.write_text(f"{dumps_prediction(record)}\n\n", encoding="utf-8")
            with self.assertRaises(SerializationError) as caught:
                read_predictions_jsonl(path)
            self.assertIn("line 2", str(caught.exception))

    def test_json_rejects_nonstandard_nan(self) -> None:
        data = make_request(num_samples=1).to_dict()
        data["observer_metadata"] = {"bad": float("nan")}
        text = json.dumps(data)
        with self.assertRaises(SerializationError):
            loads_request(text)

    def test_failed_atomic_overwrite_preserves_destination(self) -> None:
        request = make_request(num_samples=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "requests.jsonl")
            path.write_text("original\n", encoding="utf-8")
            with self.assertRaises(SerializationError):
                write_requests_jsonl(
                    path,
                    (request, object()),  # type: ignore[arg-type]
                    overwrite=True,
                )
            self.assertEqual(path.read_text(encoding="utf-8"), "original\n")


class RegistryAndDummyAdapterTests(unittest.TestCase):
    def test_registry_inspection_is_lazy_and_duplicate_is_rejected(self) -> None:
        constructions = 0

        def factory() -> DummyAdapter:
            nonlocal constructions
            constructions += 1
            return DummyAdapter()

        descriptor = AdapterDescriptor(
            model_id=DUMMY_MODEL_ID,
            identity=DUMMY_IDENTITY,
            capabilities=DUMMY_CAPABILITIES,
            requirements=dummy_descriptor().requirements,
            factory=factory,
        )
        registry = AdapterRegistry(allowed_model_ids={DUMMY_MODEL_ID})
        registry.register(descriptor)
        self.assertEqual(registry.capabilities(DUMMY_MODEL_ID), DUMMY_CAPABILITIES)
        self.assertEqual(constructions, 0)
        with self.assertRaises(DuplicateAdapterError):
            registry.register(descriptor)
        self.assertEqual(constructions, 0)
        registry.create(DUMMY_MODEL_ID).close()
        self.assertEqual(constructions, 1)

    def test_missing_optional_dependency_has_clear_error(self) -> None:
        dependency = "avrw_dependency_that_does_not_exist_7f83a6"
        descriptor = AdapterDescriptor(
            model_id=DUMMY_MODEL_ID,
            identity=DUMMY_IDENTITY,
            capabilities=DUMMY_CAPABILITIES,
            requirements=dummy_descriptor().requirements,
            factory=DummyAdapter,
            optional_dependencies=(dependency,),
        )
        registry = AdapterRegistry(allowed_model_ids={DUMMY_MODEL_ID})
        registry.register(descriptor)
        self.assertEqual(registry.capabilities(DUMMY_MODEL_ID), DUMMY_CAPABILITIES)
        with self.assertRaises(AdapterUnavailableError) as caught:
            registry.create(DUMMY_MODEL_ID)
        self.assertEqual(caught.exception.missing_dependencies, (dependency,))
        self.assertIn("will not install packages", str(caught.exception))

    def test_factory_metadata_mismatch_is_rejected(self) -> None:
        wrong_identity = ModelIdentity(
            model_id="wrong",
            model_variant="variant",
            adapter_version="1",
        )

        class WrongIdentityDummy(DummyAdapter):
            @property
            def identity(self) -> ModelIdentity:
                return wrong_identity

        descriptor = AdapterDescriptor(
            model_id=DUMMY_MODEL_ID,
            identity=DUMMY_IDENTITY,
            capabilities=DUMMY_CAPABILITIES,
            requirements=dummy_descriptor().requirements,
            factory=WrongIdentityDummy,
        )
        registry = AdapterRegistry(allowed_model_ids={DUMMY_MODEL_ID})
        registry.register(descriptor)
        with self.assertRaisesRegex(RegistryError, "identity"):
            registry.create(DUMMY_MODEL_ID)

    def test_registry_supports_multiple_variants_per_model_id(self) -> None:
        alternate_identity = ModelIdentity(
            model_id=DUMMY_MODEL_ID,
            model_variant="alternate-v1",
            adapter_version="0.1.0",
            upstream_version="builtin",
            checkpoint_id="none",
        )

        class AlternateDummy(DummyAdapter):
            @property
            def identity(self) -> ModelIdentity:
                return alternate_identity

        registry = create_dummy_registry()
        registry.register(
            AdapterDescriptor(
                model_id=DUMMY_MODEL_ID,
                identity=alternate_identity,
                capabilities=DUMMY_CAPABILITIES,
                requirements=dummy_descriptor().requirements,
                factory=AlternateDummy,
            )
        )
        self.assertEqual(
            registry.variants(DUMMY_MODEL_ID),
            ("alternate-v1", DUMMY_MODEL_VARIANT),
        )
        with self.assertRaisesRegex(RegistryError, "multiple variants"):
            registry.describe(DUMMY_MODEL_ID)
        alternate = registry.create(DUMMY_MODEL_ID, "alternate-v1")
        self.assertEqual(alternate.identity, alternate_identity)
        alternate.close()

    def test_registry_reports_unknown_and_disallowed_model_ids(self) -> None:
        registry = create_dummy_registry()
        with self.assertRaisesRegex(UnknownModelError, "available: dummy"):
            registry.describe("missing")
        disallowed = AdapterDescriptor(
            model_id="outside_scope",
            identity=ModelIdentity(
                model_id="outside_scope",
                model_variant="v1",
                adapter_version="1",
            ),
            capabilities=CapabilitySet(),
            factory=DummyAdapter,
        )
        with self.assertRaises(UnsupportedModelError):
            registry.register(disallowed)

    def test_dummy_end_to_end_lifecycle_and_determinism(self) -> None:
        request = make_request(num_samples=3)
        adapter = create_dummy_registry().create(DUMMY_MODEL_ID)
        self.assertEqual(adapter.state, AdapterState.NEW)
        self.assertFalse(adapter.is_prepared)
        first = adapter.predict(request, run_id="run-dummy")
        self.assertTrue(adapter.is_prepared)
        self.assertEqual(adapter.prepare_count, 1)  # type: ignore[attr-defined]
        adapter.prepare()
        self.assertEqual(adapter.prepare_count, 1)  # type: ignore[attr-defined]
        self.assertEqual(len(first), 3)
        self.assertEqual([record.seed for record in first], [41, 42, 43])
        self.assertTrue(
            all(record.observer_id == "observer-7" for record in first)
        )
        second = adapter.predict(request, run_id="run-dummy")
        self.assertEqual(
            [dumps_prediction(record) for record in first],
            [dumps_prediction(record) for record in second],
        )
        adapter.close()
        adapter.close()
        self.assertEqual(adapter.state, AdapterState.CLOSED)
        self.assertEqual(adapter.close_count, 1)  # type: ignore[attr-defined]
        with self.assertRaises(AdapterLifecycleError):
            adapter.predict(request, run_id="run-dummy")

    def test_dummy_requires_task_and_observer(self) -> None:
        adapter = DummyAdapter()
        with self.assertRaisesRegex(ContractValidationError, "task_text"):
            adapter.validate_request(
                make_request(task_text=None, target_description=None)
            )
        with self.assertRaisesRegex(ContractValidationError, "observer_id"):
            adapter.validate_request(
                make_request(observer_id=None, observer_metadata={})
            )
        adapter.close()

    def test_dummy_reports_actual_time_horizon_stop(self) -> None:
        request = make_request(
            num_samples=1,
            max_fixations=3,
            time_horizon_s=0.1,
        )
        with DummyAdapter() as adapter:
            record = adapter.predict(request, run_id="run-horizon")[0]
        self.assertEqual(len(record.fixations), 1)
        self.assertEqual(record.stopping_reason, StoppingReason.TIME_HORIZON)

    def test_adapter_rejects_duplicate_sample_output(self) -> None:
        class DuplicateOutputDummy(DummyAdapter):
            def _predict(
                self,
                request: InferenceRequest,
                *,
                run_id: str,
            ) -> tuple[PredictionRecord, ...]:
                records = tuple(super()._predict(request, run_id=run_id))
                return tuple(records[0] for _ in records)

        request = make_request(num_samples=3)
        adapter = DuplicateOutputDummy()
        with self.assertRaisesRegex(AdapterOutputError, "duplicate"):
            adapter.predict(request, run_id="run-duplicates")
        adapter.close()

    def test_prepare_failure_can_still_be_closed(self) -> None:
        class FailingPrepareDummy(DummyAdapter):
            def _prepare(self) -> None:
                raise RuntimeError("fixture prepare failure")

        adapter = FailingPrepareDummy()
        with self.assertRaisesRegex(RuntimeError, "fixture prepare failure"):
            adapter.prepare()
        self.assertEqual(adapter.state, AdapterState.FAILED)
        adapter.close()
        self.assertEqual(adapter.state, AdapterState.CLOSED)


class ImportTests(unittest.TestCase):
    def test_core_imports_without_site_or_heavy_model_packages(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        source = repository / "src"
        code = """
import sys
import activevision_workbench
import activevision_workbench.adapters.base
import activevision_workbench.adapters.dummy
import activevision_workbench.contracts
import activevision_workbench.registry
import activevision_workbench.serialization
forbidden = {
    'torch', 'torchvision', 'transformers', 'diffusers',
    'IndividualScanpath', 'tppgaze', 'GazeXplain', 'ScanDiff'
}
loaded = forbidden.intersection(sys.modules)
assert not loaded, sorted(loaded)
"""
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(source)
        result = subprocess.run(
            [sys.executable, "-S", "-c", code],
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
