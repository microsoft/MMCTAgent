DESCRIPTION_OD = f"""
                This tool analyzes an image to detect individual objects, returning their positions 
                in pixel coordinates using the XYHW format. XYHW represents four float values: X and Y 
                coordinates, object height, and width. Input is always empty since the tool processes 
                the provided image directly. The response is a dictionary with object labels as keys 
                and their positions as XYHW arrays. Ignore any input arguments like 'priority' and do 
                not include them in the input. you can use this only 1 time.
                """

DESCRIPTION_OCR = f"""This tool analyzes an image to extract text, useful when text information is required.
                you can use this only 1 time.
                The extracted text is returned as a list of strings, ordered from left to right and 
                top to bottom as it appears in the image. The accuracy depends on the OCR model's 
                performance and might be limited. Input is always empty since the tool directly processes 
                the image.
                The response is a list of strings representing the extracted text.
                """

DESCRIPTION_VIT = """Using this advance vision tool, you can query information about the given image/images using simple natural language,
                this tool can be used in optical character recognition, detection, description of an image, detection of objects, etc. use this model for kind of query. you can use this tool as many time as you can with diferent query.
                 e.g., "What is the color of hair in the image.", "how many object are there?"
                This returns responses in simple language.
                input: 
                    {"img_path":"image.jpeg","query": "What is the number of objects in the image"}

                    The input contain two values "query" and "img_path".
                response:
                    The output is simple text answering the query given.
                """

DESCRIPTION_RECOG = f"""You can use this tool only 1 time to analyze the given image, The tool should be used when
                you require to understand the scene in the image, and get a descriptive text
                about the image. The algorithm returns the description about the image in simple string.

                This returns response in string which is simply contains the description.
                input: 
                    
                Input is always empty as it doesnt require anything as input and analyzes on the image that you are given. Always ignore the argument priority and do not generate that in the input.

                response:
                    The output is a string containing the description.
                """

DESCRIPTION_CRITIC = f"""This tool must be called before the final response to ensure the answer meets all criteria. 
                It evaluates the response and provides feedback based on the question, image, and planner 
                agent's actions. Always use this tool, even if the question can be answered without it, but 
                only at the end of the reasoning chain. Do not use it in the middle of the chain.

                Input:
                The critic has access to the planner agent's actions, the question, and the image.

                Task:
                Call the critic before the final response. If criteria are met, provide the response. 
                Otherwise, use feedback to refine your reasoning, continue using tools, and re-evaluate 
                with the critic until all criteria are met. Ensure all criteria are satisfied before finalizing.

                Response:
                The output is text with feedback and checkboxes based on evaluation criteria.
                    """