import tempfile, os
from fastapi import HTTPException, UploadFile
from mmct.image_pipeline import ImageAgent, ImageQnaTools
from mmct.video_pipeline import VideoAgent
from loguru import logger

async def process_image_query(file: UploadFile, body: dict):
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        print(body['tools'])
        tool_list = [getattr(ImageQnaTools, t) for t in body["tools"]]
    except AttributeError:
        os.remove(tmp_path)
        raise HTTPException(400, "Invalid tool")
    agent = ImageAgent(
        query=body["query"],
        image_path=tmp_path,
        tools=tool_list,
        use_critic_agent=body["use_critic_agent"],
        stream=body["stream"],
    )
    try:
        resp = await agent()
        return resp.response
    except Exception as e:
        logger.error(e)
        raise HTTPException(500, "Image processing failed")
    finally:
        os.remove(tmp_path)


async def process_video_query(body: dict):
    agent = VideoAgent(**body)
    try:
        return await agent()
    except Exception as e:
        logger.error(e)
        raise HTTPException(500, "Video processing failed")
