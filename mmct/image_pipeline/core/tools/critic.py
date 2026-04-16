"""Critic tool for reflective feedback.

This module provides a tool that allows a critic agent to evaluate the 
reasoning and tool usage of a planner agent in an image understanding context.
"""

from typing import Annotated
from PIL import Image
from mmct.image_pipeline.core.models.vit.visual_llm import VisualLLM
from mmct.providers.base import BaseLLMProvider

class CriticTool:
    """Tool for reflective feedback on agentic reasoning.

    The CriticTool evaluates the planner's conversation history against 
    the user's query and the visual evidence in the image to ensure 
    accuracy, completeness, and logical consistency.

    Attributes:
        llm_provider (BaseLLMProvider): The LLM provider to use for evaluation.
        query (str): The original user query being analyzed.
        img_path (str): Local path to the image file.
    """

    def __init__(
        self, 
        llm_provider: BaseLLMProvider, 
        query: Annotated[str, "The original user query"], 
        img_path: Annotated[str, "Local image path"]
    ):
        """Initializes the CriticTool.

        Args:
            llm_provider: The LLM provider for vision-language evaluation.
            query: The initial question or instruction about the image.
            img_path: Local path to the image to analyze.
        """
        self.llm_provider = llm_provider
        self.query = query
        self.img_path = img_path

    async def critic_tool(
        self, 
        conversation: Annotated[str, "The full agentic conversation history"]
    ) -> str:
        """Critiques the planner's reasoning and tool usage.

        Args:
            conversation: The accumulated dialogue between agents and tool outputs.

        Returns:
            str: A structured feedback report with evaluation checkboxes and
                revisions suggestions.
        """
        prompt = f"""
                    You are a critic for a vision language pipeline, The pipeline consists
                    of a LLM comprehending a query along with image input. The LLM is able
                    to use different tools to understand the image input. It is very critical
                    to analyze 2 things, 1) Efficacy in tool usage and its performance
                    for the subtask, 2) LLMs utilization for these observation and reasoning
                    based on it.

                    For doing so you are given a the previous conversation along with main 
                    query
                    ----------------------------------------------------------------------
                    query: {self.query}
                    conversation: {conversation}
                    
                    ----------------------------------------------------------------------
                    I want a concise report which contains 4 checkboxes specified below

                    - [ ] The First checkbox denotes if the conversation has answered the
                        original query completely or even partially
                    - [ ] Understand how the tools are used and decomposed into subtasks and 
                        if They utilize all relevant information available for the query.
                        You have to take a good look into the image you are given and assert
                        if the LLM was presented with all necessary information.
                    - [ ] This is to understand any discrepancies in the reasoning chain by
                        the LLM in the conversation, You have to verify that all the steps 
                        and raise concerns if the facts are incorrect.
                    - [ ] Apart from above points if you find any other scope of improvement
                        please suggest it to the LLM. And collecting all the three points
                        finally draft a Feedback for the LLM to improve the reasoning for the
                        task.

                    You have to go through them step by step and finally format them as shown

                    - [X] Answered
                    - [ ] All information used
                    - [ ] Verification of conversation
                    - [ ] Feedback

                    The checkboxes should be filled based on the condition given above. Feedback
                    checkbox is filled when you believe that the conversation is correct in all the
                    above evaluation methods and when you cannot find any mistake in the conversation

                    In the above conversation you may see a critic verification make sure you assert 
                    those feedbacks and if they are rectified by the LLM.  
                    """
        img = Image.open(self.img_path).convert("RGB")
        model = VisualLLM(llm_provider=self.llm_provider)
        resp = await model.run(images=img,prompt=prompt)
        return resp