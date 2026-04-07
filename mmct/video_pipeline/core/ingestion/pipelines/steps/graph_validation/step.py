"""Graph validation step: skip ingestion if video already exists in Neo4j."""

from ..base import PipelineStep, StepContext, StepResult
from ..registry import register_step

from loguru import logger


@register_step("ingestion.graph_validation")
class GraphValidationStep(PipelineStep):
    """
    Check the graph store for existing ChapterGroup and Chapter nodes
    for the given video_id. If both are present, signal the pipeline to
    stop (should_continue=False) so already-ingested videos are skipped.

    Params:
        require_chapter_groups: Also require ChapterGroup nodes (default: True).
            When False, only Chapter nodes are checked.
    """

    step_type = "ingestion.graph_validation"
    description = "Check graph store for existing video data"

    async def run(self, context: StepContext) -> StepResult:
        video_id = context.video_id
        require_groups = self.get_param("require_chapter_groups", context, default=True)

        graph_store = getattr(context.provider, "graph_store_provider", None)
        if graph_store is None:
            context.logger.warning(
                "No graph_store_provider configured — skipping graph validation"
            )
            return StepResult(
                step_id=self.step_id,
                outputs={"graph_exists": False, "should_continue": True},
            )

        context.logger.debug(
            f"Checking graph store for existing data (video_id={video_id})"
        )

        try:
            stats = await graph_store.get_video_stats(video_id)
        except Exception as e:
            context.logger.warning(f"Graph store query failed: {e} — proceeding with ingestion")
            return StepResult(
                step_id=self.step_id,
                outputs={"graph_exists": False, "should_continue": True},
            )

        node_counts = stats.get("node_counts", {})
        chapter_count = node_counts.get("Chapter", 0)
        group_count = node_counts.get("ChapterGroup", 0)

        has_chapters = chapter_count > 0
        has_groups = group_count > 0

        if has_chapters and (has_groups or not require_groups):
            context.logger.info(
                f"Video {video_id} already in graph store "
                f"(Chapters={chapter_count}, ChapterGroups={group_count}). "
                f"Skipping ingestion."
            )
            return StepResult(
                step_id=self.step_id,
                outputs={
                    "graph_exists": True,
                    "chapter_count": chapter_count,
                    "chapter_group_count": group_count,
                    "should_continue": False,
                },
            )

        context.logger.debug(
            f"Video {video_id} not found in graph store "
            f"(Chapters={chapter_count}, ChapterGroups={group_count}). "
            f"Proceeding with ingestion."
        )
        return StepResult(
            step_id=self.step_id,
            outputs={
                "graph_exists": False,
                "chapter_count": chapter_count,
                "chapter_group_count": group_count,
                "should_continue": True,
            },
        )
