"""
This file contains description of the tools.
"""

DESC_get_transcript = """This tool returns the full transcript of the video along with timestamps for each phrase. This tool should be called first. This would allow you to directly tackle the questions that are answerable by just looking at the transcript modality. If this is the case, just answer and stop there and do not unnecessarily call other tools. If not, in many cases, the transcript might contain a partial answer, a related event, or any hint/reference indicating where in the visuals the answer might be found. If that is the case then you must diligently note down these details from the transcript in your "observation" and remember them for future use since they will help you in deciding whether to retrieve potentially relevant visuals using query_transcript or not. However, if neither of these are true, then looking at the transcript would still give you a basic understanding of the video and might enable you to answer some generic questions like video summary and also dismissing extremely irrelevant questions. In case the transcript is empty, you must understand that this video only contains visuals and hence focus only on that."""

DESC_get_summary = """This tool gives you the detailed summary, action_taken, transript of the video. This tool should be called first in the conversation chain. This would allow you to directly tackle the questions that are answerable by just looking at the detailed summary, action_taken, transcript."""

DESC_query_GPT4_Vision = """This tool is designed to allow you to verify the retrieved timestamps from other tools and also ask more nuanced questions about these localized segments of the video. 
                        It utilizes GPT4's Vision capabilities and passes a 10 second clip (only visuals, no audio or transcript) sampled at 1 fps and centered at 'timestamp' 
                        (which is likely returned by other tools; its format is the same i.e. %H:%M:%S) along with a 'query' to the model. Note that this query can be any prompt designed to extract the required information regarding the clip in consideration. 
                        The output is simply GPT4's response to the given clip and prompt.
                        """

DESC_query_transcript = """This tool allows you to issue a search query over the video transcript and return the timestamps of the top 3 semantically matched phrases in the transcript. 
                            The returned timestamps are the average time between the start and end of matched phrases. The timestamps would be comma separated (presented in their matching order with the leftmost being the highest match) 
                            and in the format %H:%M:%S (e.g. 00:08:27, 00:23:56, 01:14:39)
                            """

DESC_query_summary = """This tool allows you to issue a search query over the video summary and return the timestamps of the top 3 semantically matched phrases in the summary. 
                            The returned timestamps are the average time between the start and end of matched phrases. The timestamps would be comma separated (presented in their matching order with the leftmost being the highest match) 
                            and in the format %H:%M:%S (e.g. 00:08:27, 00:23:56, 01:14:39)
                            """

DESC_query_frames_Azure_Computer_Vision = """This tool allows you to issue a natural language search query over the frames of the video using Azure's Computer Vision API to a find a specific moment in the video. 
                                        It is good at OCR, object detection and much more. The output format is similar to the query_transcript tool. It returns comma separated timestamps of the top 3 frames that match with given query. do not repeat the same search query.
                                    """

DESC_criticTool = """this tool allows you to criticise the planner response. this agent can only participate in the conversation when explicity asked by planner agent.
                    """



