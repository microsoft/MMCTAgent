"""
Test script for pipeline framework - follows GitHub repo pattern.

Run from project root:
    python -m mmct.video_pipeline.core.ingestion.scripts.test_framework
"""

import sys
import os
import asyncio
from loguru import logger

from mmct.video_pipeline.core.ingestion.pipelines.steps import (
    PipelineStep,
    StepContext,
    StepResult,
    register_step,
    get_step_class,
    available_steps,
    StepDataStore,
)
from mmct.video_pipeline.core.ingestion.pipelines.runner import PipelineRunner
from mmct.video_pipeline.core.ingestion.pipelines.config.schemas import PipelineConfig, StepConfig, load_pipeline_config


def print_section(title):
    """Print section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_step_registration():
    """Verify all steps are registered."""
    print_section("TEST 1: Step Registration")

    steps = available_steps()
    print(f"\n✓ Registered steps: {len(steps)}")

    for step_type in sorted(steps):
        try:
            step_class = get_step_class(step_type)
            print(f"  ✓ {step_type:40} -> {step_class.__name__}")
        except Exception as e:
            print(f"  ✗ {step_type:40} -> ERROR: {e}")
            return False

    return True


def test_config_loading():
    """Test YAML configuration loading."""
    print_section("TEST 2: Configuration Loading")

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "experiments/default_ingestion.yaml"
    )

    if not os.path.exists(config_path):
        print(f"\n✗ Config not found: {config_path}")
        return False

    try:
        config = load_pipeline_config(config_path)
        print(f"\n✓ Pipeline: {config.name}")
        print(f"✓ Mode: {config.mode}")
        print(f"✓ Steps: {len(config.steps)}")

        print("\nPipeline flow:")
        for i, step in enumerate(config.steps, 1):
            print(f"  {i:2}. {step.id:20} ({step.type})")

        return True
    except Exception as e:
        print(f"\n✗ Config loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_store():
    """Test StepDataStore."""
    print_section("TEST 3: Data Store")

    store = StepDataStore()

    store.set("step1", "output", "value1")
    store.set("step2", "result", {"key": "value"})

    assert store.get("step1", "output") == "value1"
    assert store.has("step1")
    assert not store.has("nonexistent")

    print("\n✓ Data store operations verified")
    return True


async def test_mock_pipeline():
    """Test pipeline execution with mock step."""
    print_section("TEST 4: Mock Pipeline Execution")

    # Create mock step
    @register_step("test.mock")
    class MockStep(PipelineStep):
        step_type = "test.mock"
        description = "Mock step for testing"

        async def run(self, context: StepContext) -> StepResult:
            context.logger.info(f"Executing {self.step_id}...")
            return StepResult(
                step_id=self.step_id,
                outputs={"result": f"output_{self.step_id}"},
                metrics={"time": 0.1},
                artifacts=[]
            )

    # Create simple pipeline
    config = PipelineConfig(
        name="test",
        mode="sequential",
        steps=[
            StepConfig(id="step1", type="test.mock", params={}),
            StepConfig(id="step2", type="test.mock", params={}),
            StepConfig(id="step3", type="test.mock", params={}),
        ]
    )

    # Create context
    context = StepContext(
        video_path="/mock/video.mp4",
        provider=None,
        output_dir="/tmp",
        video_id="test123",
        video_duration=100.0,
        data_store=StepDataStore(),
        logger=logger,
        user_params={}
    )

    # Run
    try:
        runner = PipelineRunner(pipeline_config=config, context=context)
        report = await runner.run()

        print(f"\n✓ Pipeline: {report.status}")
        print(f"✓ Duration: {report.total_duration_seconds:.2f}s")
        print(f"✓ Steps executed: {len(report.steps)}")

        for rec in report.steps:
            print(f"    {rec.step_id}: {rec.status}")

        return True
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print_section("PIPELINE FRAMEWORK TEST SUITE")

    tests = [
        ("Step Registration", test_step_registration),
        ("Configuration Loading", test_config_loading),
        ("Data Store", test_data_store),
        ("Mock Pipeline", test_mock_pipeline),
    ]

    results = []
    for name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print_section("SUMMARY")
    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status:10} {name}")

    print(f"\n  Result: {passed}/{total} tests passed\n")

    return 0 if passed == total else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
