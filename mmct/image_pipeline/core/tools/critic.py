"""
This is a vit tool. it uses the GPT4V.
"""

from mmct.image_pipeline.core.models.vit.gpt4v import GPT4V
from PIL import Image
from typing_extensions import Annotated

async def critic_tool(img_path: Annotated[str, "path of image"], query: Annotated[str, "query about the image"], conversation:Annotated[str,"all past conversation between agents."]) -> str:
    """
    critic tool for critic agent which criticise the planner response.
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
                query: {query}
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
    img = Image.open(img_path).convert("RGB")
    a = GPT4V()
    resp = await a.run(images=img,prompt=prompt)
    return resp